import collections
import dataclasses
import glob
import json
import os
import shutil
import time
from typing import NamedTuple

import numpy as np

import fileio
from counting import (
    MAX_TABLE_COLUMNS,
    MAX_UINT64_COLUMNS,
    CountSettings,
    add_counts,
    count_table,
    encode_tokens,
    load_checkpoint,
    measure_overlap,
    select_rows,
    vocabulary_pass,
    word_masks,
    words_hash,
)
from fetch import add_time_used, load_sources
from model import Model, ModelError, evaluate, hub_scores, load_model, save_model
from puzzlebuild import is_word, zipf_of
from scoring import (
    file_hash,
    load_probes,
    load_tagged_puzzles,
    puzzle_results,
    score_report,
)
from textread import (
    Cleaning,
    group_order,
    list_groups,
    read_text_file,
    tokenize_text,
)
from vectors import VectorSettings, embed, no_svd_vectors, ppmi_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FULL_WORDS = 8200000000
MAX_ROWS = 100000
SWEEP_WORDS = (100000000, 1000000000)
PASS_AT = 0.90
HUB_TRIGGER = 0.80
OVERLAP_LIMIT = 0.01
COUNT_GRID = [
    ("stoplist", ["base30", "full", "top100"]),
    ("window", [2, 5, 10]),
    ("n_columns", [3000, 5000, 10000]),
    ("directional", [False, True]),
    ("breaks", ["line", "sentence"]),
    ("apostrophes", ["split", "join"]),
    ("accents", ["split", "fold"]),
    ("subsample", [None, 1e-5]),
]
VECTOR_GRID = [("p", [0.0, 0.5, 1.0]), ("dims", [200, 300]), ("centre", [True, False])]
SMALL_DIMS = 300
METHOD_GRID = [
    ("p", [-0.5, -0.25]),
    ("skip", [1, 2, 5, 10]),
    ("shift", [2.0, 5.0]),
    ("smoothing", [0.5, 1.0]),
    ("dims", [400, 500, 600]),
    ("rule", ["median", "max"]),
    ("p", [-0.5, -0.25, 0.0]),
]
VECTOR_FIELDS = {field.name for field in dataclasses.fields(VectorSettings)}
CLEANING_FIELDS = {field.name for field in dataclasses.fields(Cleaning)}


class TrainError(Exception):
    pass


class Paths(NamedTuple):
    text: str
    tune: str
    dev: str
    probes: str
    sources: str | None = None


def default_paths():
    return Paths(
        os.path.join(DATA, "raw", "text"),
        os.path.join(DATA, "puzzles_tune.txt"),
        os.path.join(DATA, "puzzles_diagnostic.txt"),
        os.path.join(DATA, "probes.txt"),
        os.path.join(HERE, "sources.json"),
    )


def default_workers():
    return max(1, min(12, (os.cpu_count() or 2) - 2))


def freeze_decision(tune_score):
    return "freeze" if tune_score >= PASS_AT else "stall"


def ship_milestone(curve, range_width):
    full = max(curve)
    smaller = [words for words in curve if words != full]
    if not smaller:
        return full
    best = max(smaller, key=lambda words: (curve[words], words))
    return best if curve[best] - curve[full] > range_width else full


def better(new, old, gain=0.005):
    return new - old >= gain


def change_settings(settings, name, value):
    if name in CLEANING_FIELDS:
        cleaning = dataclasses.replace(settings.cleaning, **{name: value})
        return dataclasses.replace(settings, cleaning=cleaning)
    return dataclasses.replace(settings, **{name: value})


def current_value(settings, name):
    holder = settings.cleaning if name in CLEANING_FIELDS else settings
    return getattr(holder, name)


def fit_table(settings):
    if settings.directional and settings.table_columns() > MAX_UINT64_COLUMNS:
        return dataclasses.replace(settings, n_columns=MAX_UINT64_COLUMNS // 2)
    return settings


def vector_key(vector):
    fields = dataclasses.asdict(vector)
    for name in ("shift", "skip", "rule"):
        if fields[name] == getattr(VectorSettings(), name):
            del fields[name]
    return json.dumps(fields, sort_keys=True)


def run_key(settings, vector, words):
    return f"{settings.key()}_{vector_key(vector)}_{words}"


def read_tune_rows(log_path):
    if log_path is None or not os.path.exists(log_path):
        return []
    return [json.loads(line) for line in fileio.read_lines(log_path)]


def append_row(log_path, row):
    if log_path is not None:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        fileio.append_line(log_path, json.dumps(row))


def sweep_settings(
    measure_fn,
    budget_seconds,
    clock=time.monotonic,
    log_path=None,
    start=None,
    words=SWEEP_WORDS,
    tag="",
):
    settings, vector = start or (CountSettings(), VectorSettings())
    began = clock()
    done = {r["key"]: r for r in read_tune_rows(log_path) if "key" in r}
    rates = {"word": 0.0, "vector": 0.0}

    def measure(s, v, n, vector_only=False):
        key = f"{tag}{run_key(s, v, n)}"
        result = done.get(key)
        if result is None:
            estimate = rates["word"] * n
            if vector_only and rates["vector"]:
                estimate = rates["vector"]
            reserve = 2 * rates["word"] * FULL_WORDS
            if clock() - began + estimate + reserve >= budget_seconds:
                return None
            result = measure_fn(s, v, n)
            row = {
                "key": key,
                "settings": dataclasses.asdict(s),
                "vector": dataclasses.asdict(v),
                "words": n,
                **result,
            }
            done[key] = row
            append_row(log_path, row)
        if vector_only:
            rates["vector"] = result["seconds"]
        else:
            rates["word"] = max(rates["word"], result["seconds"] / n)
        return result["tune"]

    def sweep_axis(name, values, s, score, n):
        for value in values:
            if value == current_value(s, name):
                continue
            candidate = fit_table(change_settings(s, name, value))
            if candidate == s or candidate.table_columns() > MAX_UINT64_COLUMNS:
                continue
            found = measure(candidate, vector, n)
            if found is not None and better(found, score):
                s, score = candidate, found
        return s, score

    small, large = words
    score = measure(settings, vector, small)
    if score is None:
        return settings, vector
    for name, values in COUNT_GRID:
        settings, score = sweep_axis(name, values, settings, score, small)
    score = measure(settings, vector, large)
    if score is None:
        return settings, vector
    grid = dict(COUNT_GRID)
    for name in ("window", "n_columns"):
        settings, score = sweep_axis(name, grid.get(name, []), settings, score, large)
    for name, values in VECTOR_GRID:
        for value in values:
            if value == getattr(vector, name):
                continue
            candidate = dataclasses.replace(vector, **{name: value})
            found = measure(settings, candidate, large, vector_only=True)
            if found is not None and better(found, score):
                vector, score = candidate, found
    return settings, vector


class State(NamedTuple):
    settings: CountSettings
    vector: VectorSettings
    result: dict | None


class Round(NamedTuple):
    best: State
    small: State | None
    history: list
    baseline: dict | None


def tune_folds(targets, seed=780):
    names = sorted(set(targets))
    order = np.random.default_rng(seed).permutation(len(names))
    fold_of = {names[i]: place % 2 for place, i in enumerate(order)}
    return [fold_of[target] for target in targets]


def fold_scores(results, folds):
    scores = []
    for fold in (0, 1):
        chosen = [result for result, f in zip(results, folds) if f == fold]
        scores.append(sum(chosen) / len(chosen) if chosen else 0.0)
    return scores


def keeps(new, old, gain=0.005):
    return better(new["tune"], old["tune"], gain) and all(
        n >= o for n, o in zip(new["folds"], old["folds"])
    )


def method_round(
    measure_fn,
    start,
    words,
    grid=METHOD_GRID,
    log_path=None,
    tag="",
    budget_seconds=float("inf"),
    clock=time.monotonic,
):
    settings, vector = start
    began = clock()
    done = {
        r["key"]: r for r in read_tune_rows(log_path) if "key" in r and "folds" in r
    }
    last = 0.0

    def measure(s, v):
        nonlocal last
        key = f"{tag}{run_key(s, v, words)}"
        result = done.get(key)
        if result is None:
            if clock() - began + last >= budget_seconds:
                return None
            started = clock()
            result = measure_fn(s, v, words)
            last = clock() - started
            row = {
                "key": key,
                "settings": dataclasses.asdict(s),
                "vector": dataclasses.asdict(v),
                "words": words,
                **result,
            }
            done[key] = row
            append_row(log_path, row)
        return result

    baseline = measure(settings, vector)
    if baseline is None:
        return Round(State(settings, vector, None), None, [], None)
    best = State(settings, vector, baseline)
    small = best if vector.dims <= SMALL_DIMS else None
    history = []
    for name, values in grid:
        for value in values:
            if name in VECTOR_FIELDS:
                if value == getattr(best.vector, name):
                    continue
                s = best.settings
                v = dataclasses.replace(best.vector, **{name: value})
            else:
                if value == current_value(best.settings, name):
                    continue
                s = change_settings(best.settings, name, value)
                v = best.vector
                if s.table_columns() > MAX_TABLE_COLUMNS:
                    continue
            result = measure(s, v)
            if result is None:
                return Round(best, small, history, baseline)
            kept = keeps(result, best.result)
            history.append(
                {
                    "name": name,
                    "value": value,
                    "tune": result["tune"],
                    "folds": result["folds"],
                    "kept": kept,
                }
            )
            if kept:
                best = State(s, v, result)
                if v.dims <= SMALL_DIMS:
                    small = best
    return Round(best, small, history, baseline)


def pick_finalist(best, small, stored):
    if best.vector.dims <= SMALL_DIMS or small is None:
        return best
    return best if stored >= small.result["tune"] else small


def bits_for(vector):
    return 16 if vector.dims > SMALL_DIMS else 32


def scores_of(report_part):
    return {name: item["score"] for name, item in report_part.items()}


def puzzle_words(puzzle_sets):
    words = set()
    for items in puzzle_sets:
        for puzzle in items:
            words.update(puzzle[0])
    return sorted(words)


class Runner:
    def __init__(self, jobs, paths, work_dir, workers, skip_urls=None, no_svd_at=None):
        self.jobs = jobs
        self.work_dir = work_dir
        self.workers = workers
        self.skip_urls = skip_urls
        self.no_svd_at = no_svd_at
        self.tune = load_tagged_puzzles(paths.tune)
        self.dev = load_dev(paths.dev)
        self.probes = load_probes(paths.probes)
        self.folds = tune_folds([p.tags["target"] for p in self.tune])
        self.sweep_dir = os.path.join(work_dir, "sweep")
        self.counts_dir = None
        self.kept = None

    def vocabulary(self, cleaning, words):
        name = json.dumps(dataclasses.asdict(cleaning), sort_keys=True)
        name = words_hash([name, str(words), str(self.jobs)])
        path = os.path.join(self.work_dir, f"vocab_{name}.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                saved = json.load(f)
            used = [tuple(job) for job in saved["jobs"]]
            return saved["counts"], used, saved["total"]
        counts, used, total = vocabulary_pass(
            self.jobs,
            cleaning,
            self.workers,
            max_words=words,
            skip_urls=self.skip_urls,
        )
        counts = {word: counts[word] for word in select_rows(counts, MAX_ROWS)}
        os.makedirs(self.work_dir, exist_ok=True)
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            json.dump({"counts": counts, "jobs": used, "total": total}, f)
        os.replace(path + ".tmp", path)
        return counts, used, total

    def matrix(self, settings, smoothing, shift, words):
        key = (settings.key(), smoothing, shift, words)
        if self.kept is not None and self.kept[0] == key:
            return self.kept[1], self.kept[2]
        self.kept = None
        counts, used, _ = self.vocabulary(settings.cleaning, words)
        rows = select_rows(counts)
        table = count_table(
            used,
            rows,
            settings,
            self.workers,
            [],
            self.counts_dir or self.sweep_dir,
            None,
            self.skip_urls,
            counts,
        )
        matrix = ppmi_matrix(table, smoothing, shift)
        del table
        if self.counts_dir is None:
            shutil.rmtree(self.sweep_dir, ignore_errors=True)
        self.kept = (key, rows, matrix)
        return rows, matrix

    def score(self, rows, matrix, vector):
        vectors = embed(matrix, vector)
        result, report = self.rate(rows, vectors, None, vector.rule)
        return result, report, vectors

    def rate(self, rows, vectors, hub=None, rule="sum"):
        index = {word: i for i, word in enumerate(rows)}
        model = Model(rows, index, vectors, {"rule": rule}, hub)
        report = score_report(self.tune, model)
        folds = fold_scores(puzzle_results(self.tune, model), self.folds)
        result = {
            "tune": report["overall"]["score"],
            "folds": folds,
            "by_part": scores_of(report["by_part"]),
            "by_type": scores_of(report["by_type"]),
            "dev": evaluate(self.dev, model),
            "probes": {t: evaluate(p, model) for t, p in self.probes.items()},
            "n_rows": len(rows),
        }
        return result, report

    def no_svd(self, rows, matrix):
        index = {word: i for i, word in enumerate(rows)}
        wanted = puzzle_words(
            [[(p.words, p.answer) for p in self.tune], self.dev, *self.probes.values()]
        )
        wanted = [word for word in wanted if word in index]
        vectors = no_svd_vectors(matrix, [index[word] for word in wanted])
        model = Model(wanted, {w: i for i, w in enumerate(wanted)}, vectors, {})
        report = score_report(self.tune, model)
        return {
            "tune": report["overall"]["score"],
            "by_part": scores_of(report["by_part"]),
            "by_type": scores_of(report["by_type"]),
            "dev": evaluate(self.dev, model),
        }

    def __call__(self, settings, vector, words):
        began = time.monotonic()
        rows, matrix = self.matrix(settings, vector.smoothing, vector.shift, words)
        result, _, _ = self.score(rows, matrix, vector)
        if self.no_svd_at is not None and words == self.no_svd_at:
            result["no_svd"] = self.no_svd(rows, matrix)
        result["seconds"] = time.monotonic() - began
        record_time(self.work_dir, result["seconds"])
        return result


def load_dev(path):
    from cli import load_puzzles

    return load_puzzles(path)


def record_time(work_dir, seconds):
    add_time_used(os.path.join(work_dir, "..", "raw"), "train", seconds)


def time_used(work_dir):
    path = os.path.join(work_dir, "time_used.json")
    if not os.path.exists(path):
        return 0.0
    with open(path, encoding="utf-8") as f:
        return sum(json.load(f).values())


def text_files(paths):
    uk = sorted(glob.glob(os.path.join(paths.text, "uk", "*.parquet")))
    fine = sorted(glob.glob(os.path.join(paths.text, "fine", "*.parquet")))
    if not uk or not fine:
        raise TrainError(f"No text in {paths.text}. Run:  python cli.py download")
    if paths.sources is not None:
        expected = {
            os.path.join(paths.text, entry["group"], entry["name"]): entry["size"]
            for entry in load_sources(paths.sources)["text"]
        }
        if {path: os.path.getsize(path) for path in uk + fine} != expected:
            raise TrainError(
                f"The text in {paths.text} is not complete. "
                "Run:  python cli.py download"
            )
    return uk, fine


def check_files(paths):
    if not os.path.exists(paths.tune):
        raise TrainError(f"{paths.tune} is missing. Run:  python cli.py puzzles")


def load_overlap(work_dir, uk, fine, workers):
    path = os.path.join(work_dir, "overlap.json")
    hashes_path = os.path.join(work_dir, "overlap_hashes.npy")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            found = json.load(f)
        hashes = set(np.load(hashes_path).tolist())
    else:
        found = measure_overlap(uk, fine, workers)
        hashes = found.pop("skip_urls")["hashes"]
        os.makedirs(work_dir, exist_ok=True)
        np.save(hashes_path, np.array(sorted(hashes), dtype=np.uint64))
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(found, f)
        os.replace(path + ".tmp", path)
    skip = None
    if found["shared_token_share"] > OVERLAP_LIMIT:
        skip = {"hashes": hashes, "paths": set(fine)}
    return found, skip


def zipf3_coverage(words):
    from wordfreq import top_n_list

    listed = [
        word
        for word in top_n_list("en", 200000, wordlist="large")
        if is_word(word) and zipf_of(word) >= 3
    ]
    known = set(words)
    return sum(word in known for word in listed) / len(listed)


def print_outcome(result):
    print("Learning curve:")
    for words, score in sorted(result["curve"].items()):
        print(f"  {words:>14,} words  tune {score:.3f}")
    print(f"Shipped at {result['shipped']:,} words, tune {result['score']:.3f}")
    print("By part:", result["by_part"])
    print("By type:", result["by_type"])
    print("No-SVD diagnostic:", result["no_svd"])
    print("Hubness correction:", result["hubness"])
    print("Decision:", result["decision"])


def train_full(
    work_dir,
    model_dir,
    budget_hours=48,
    paths=None,
    workers=None,
    sweep_words=SWEEP_WORDS,
    milestones=SWEEP_WORDS,
    start_settings=None,
    start_vector=None,
):
    paths = paths or default_paths()
    workers = default_workers() if workers is None else workers
    uk, fine = text_files(paths)
    check_files(paths)
    jobs = group_order(list_groups(uk), list_groups(fine))
    overlap, skip_urls = load_overlap(work_dir, uk, fine, workers)
    log_path = os.path.join(work_dir, "tune_log.jsonl")
    runner = Runner(jobs, paths, work_dir, workers, skip_urls, sweep_words[-1])
    budget = budget_hours * 3600.0
    start = (start_settings or CountSettings(), start_vector or VectorSettings())
    tune_hash = file_hash(paths.tune)
    settings, vector = sweep_settings(
        runner,
        budget - time_used(work_dir),
        log_path=log_path,
        start=start,
        words=sweep_words,
        tag=tune_hash,
    )
    runner.kept = None

    began = time.monotonic()
    counts, used, total = runner.vocabulary(settings.cleaning, None)
    rows = select_rows(counts)
    hash_of_words = words_hash(rows)
    checks = os.path.join(work_dir, "checks")
    full_key = f"{tune_hash}{run_key(settings, vector, 'full')}_{hash_of_words}"
    curve, checkpoint_of, no_svd = {}, {}, None
    for row in read_tune_rows(log_path):
        if row.get("full_key") == full_key:
            curve[row["words"]] = row["tune"]
            checkpoint_of[row["words"]] = row["milestone"]
            no_svd = row.get("no_svd", no_svd)
    done_names = set(checkpoint_of.values())

    def on_milestone(milestone, seen, table):
        nonlocal no_svd
        if milestone in done_names:
            return
        matrix = ppmi_matrix(table, vector.smoothing, vector.shift)
        result, _, _ = runner.score(rows, matrix, vector)
        row = {"full_key": full_key, "milestone": milestone, "words": seen, **result}
        if milestone == max(milestones):
            no_svd = runner.no_svd(rows, matrix)
            row["no_svd"] = no_svd
        append_row(log_path, row)
        curve[seen] = result["tune"]
        checkpoint_of[seen] = milestone

    table = count_table(
        used,
        rows,
        settings,
        workers,
        list(milestones),
        checks,
        on_milestone,
        skip_urls,
        counts,
    )
    matrix = ppmi_matrix(table, vector.smoothing, vector.shift)
    del table
    result, report, vectors = runner.score(rows, matrix, vector)
    curve[total] = result["tune"]
    checkpoint_of[total] = "final"
    if no_svd is None:
        no_svd = runner.no_svd(rows, matrix)
    append_row(
        log_path,
        {
            "full_key": full_key,
            "milestone": "final",
            "words": total,
            "no_svd": no_svd,
            **result,
        },
    )
    low, high = report["overall"]["bootstrap"]
    shipped = ship_milestone(curve, high - low)
    del matrix
    if shipped != total:
        table = load_checkpoint(checks, settings, hash_of_words, checkpoint_of[shipped])
        matrix = ppmi_matrix(table, vector.smoothing, vector.shift)
        del table
        result, report, vectors = runner.score(rows, matrix, vector)
    hub, hubness = None, None
    if result["by_type"].get("close", 1.0) < HUB_TRIGGER:
        hub = hub_scores(vectors)
        tried, tried_report = runner.rate(rows, vectors, hub, vector.rule)
        kept = better(tried["tune"], result["tune"])
        hubness = {
            "tune_before": result["tune"],
            "tune_after": tried["tune"],
            "close_before": result["by_type"]["close"],
            "close_after": tried["by_type"]["close"],
            "kept": kept,
        }
        append_row(log_path, {"hubness": hubness})
        if kept:
            result, report = tried, tried_report
        else:
            hub = None
    record_time(work_dir, time.monotonic() - began)
    decision = freeze_decision(result["tune"])
    meta = {
        "settings": dataclasses.asdict(settings),
        "vector": dataclasses.asdict(vector),
        "rule": vector.rule,
        "curve": curve,
        "shipped_words": shipped,
        "tune_report": report,
        "no_svd": no_svd,
        "hubness": hubness,
        "frozen": True,
        "stalled": decision == "stall",
        "coverage": zipf3_coverage(rows),
        "overlap": overlap,
        "dedupe": skip_urls is not None,
    }
    save_model(model_dir, rows, vectors, meta, hub)
    outcome = {
        "decision": decision,
        "score": result["tune"],
        "curve": curve,
        "shipped": shipped,
        "by_part": result["by_part"],
        "by_type": result["by_type"],
        "no_svd": no_svd,
        "hubness": hubness,
    }
    print_outcome(outcome)
    return outcome


def saved_settings(saved):
    fields = dict(saved)
    cleaning = Cleaning(**fields.pop("cleaning"))
    return CountSettings(**fields, cleaning=cleaning)


def store_candidate(runner, state, folder):
    began = time.monotonic()
    rows, matrix = runner.matrix(
        state.settings, state.vector.smoothing, state.vector.shift, None
    )
    vectors = embed(matrix, state.vector)
    del matrix
    meta = {
        "settings": dataclasses.asdict(state.settings),
        "vector": dataclasses.asdict(state.vector),
        "rule": state.vector.rule,
        "bits": bits_for(state.vector),
    }
    save_model(folder, rows, vectors, meta)
    stored = load_model(folder)
    result, report = runner.rate(rows, stored.vectors, None, state.vector.rule)
    record_time(runner.work_dir, time.monotonic() - began)
    return rows, np.array(stored.vectors), result, report


def print_round(outcome):
    print(
        f"Tune score before {outcome['before']:.4f}, candidate {outcome['score']:.4f}"
    )
    print("Comparisons:")
    for step in outcome["history"]:
        kept = "kept" if step["kept"] else "rejected"
        folds = ", ".join(f"{fold:.3f}" for fold in step["folds"])
        print(
            f"  {step['name']}={step['value']}  tune {step['tune']:.4f}"
            f"  folds {folds}  {kept}"
        )
    print("By part:", outcome["by_part"])
    print("By type:", outcome["by_type"])
    print(f"Stored in {outcome['bits']} bits")
    print(
        "Model replaced:" if outcome["replaced"] else "Model kept:", outcome["decision"]
    )


def run_method_round(
    work_dir,
    model_dir,
    candidate_dir=None,
    budget_hours=12,
    paths=None,
    workers=None,
    start_settings=None,
    start_vector=None,
    grid=METHOD_GRID,
):
    try:
        saved = load_model(model_dir)
    except ModelError as e:
        raise TrainError(f"No saved model in {model_dir}. Run train first.") from e
    old_meta = saved.meta
    del saved
    if "tune_report" not in old_meta:
        raise TrainError(
            f"The model in {model_dir} has no tune report. Run train first."
        )
    candidate_dir = candidate_dir or os.path.join(work_dir, "model_candidate")
    paths = paths or default_paths()
    workers = default_workers() if workers is None else workers
    uk, fine = text_files(paths)
    check_files(paths)
    jobs = group_order(list_groups(uk), list_groups(fine))
    _, skip_urls = load_overlap(work_dir, uk, fine, workers)
    runner = Runner(jobs, paths, work_dir, workers, skip_urls)
    runner.counts_dir = os.path.join(work_dir, "checks")
    start = (
        start_settings or saved_settings(old_meta["settings"]),
        start_vector or VectorSettings(**old_meta["vector"]),
    )
    found = method_round(
        lambda s, v, words: runner(s, v, None),
        start,
        None,
        grid,
        os.path.join(work_dir, "tune_log.jsonl"),
        file_hash(paths.tune),
        budget_hours * 3600.0,
    )
    if found.best.result is None:
        raise TrainError("The time budget left no time to measure the saved model.")
    final = found.best
    rows, vectors, result, report = store_candidate(runner, final, candidate_dir)
    chosen = pick_finalist(final, found.small, result["tune"])
    if chosen is not final:
        final = chosen
        rows, vectors, result, report = store_candidate(runner, final, candidate_dir)
    before = old_meta["tune_report"]["overall"]["score"]
    replaced = better(result["tune"], before)
    decision = freeze_decision(result["tune"] if replaced else before)
    if replaced:
        meta = {
            **old_meta,
            "settings": dataclasses.asdict(final.settings),
            "vector": dataclasses.asdict(final.vector),
            "rule": final.vector.rule,
            "bits": bits_for(final.vector),
            "hubness": None,
            "frozen": True,
            "stalled": decision == "stall",
            "tune_report": report,
            "method_round": {
                "baseline": found.baseline["tune"],
                "history": found.history,
                "folds": result["folds"],
                "measurements": len(found.history) + 1,
            },
        }
        save_model(model_dir, rows, vectors, meta)
    outcome = {
        "decision": decision,
        "score": result["tune"],
        "before": before,
        "history": found.history,
        "bits": bits_for(final.vector),
        "replaced": replaced,
        "by_part": result["by_part"],
        "by_type": result["by_type"],
    }
    print_round(outcome)
    return outcome


def train_corpus(paths, settings, vector_settings, out):
    cleaning = dataclasses.replace(
        settings.cleaning, breaks="sentence", min_line_words=0
    )
    texts = [tokenize_text(read_text_file(path), cleaning) for path in paths]
    counts = collections.Counter()
    for lines in texts:
        for line in lines:
            counts.update(w for w in line if len(w) <= cleaning.max_word_len)
    rows = select_rows(counts)
    index = {word: i for i, word in enumerate(rows)}
    is_stop, col_of = word_masks(rows, settings)
    table = np.zeros((len(rows), settings.table_columns()), dtype=np.uint64)
    for lines in texts:
        seq = encode_tokens(lines, index)
        add_counts(table, seq, is_stop, col_of, settings.window, settings.directional)
    matrix = ppmi_matrix(table, vector_settings.smoothing, vector_settings.shift)
    vectors = embed(matrix, vector_settings)
    meta = {
        "settings": dataclasses.asdict(settings),
        "vector": dataclasses.asdict(vector_settings),
        "rule": vector_settings.rule,
        "frozen": False,
        "stalled": False,
    }
    save_model(out, rows, vectors, meta)
    model = load_model(out)
    defaults = default_paths()
    print(f"Saved {len(rows)} words to {out}")
    print(f"dev score {evaluate(load_dev(defaults.dev), model):.3f}")
    for tier, items in load_probes(defaults.probes).items():
        print(f"probe {tier} {evaluate(items, model):.3f}")
    return model


def peak_ram_gb():
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    except ImportError:
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("faults", wintypes.DWORD),
                ("peak", ctypes.c_size_t),
                ("rest", ctypes.c_size_t * 7),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        psapi = ctypes.WinDLL("psapi")
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(Counters),
            wintypes.DWORD,
        ]
        handle = wintypes.HANDLE(-1)
        psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        return counters.peak / 1024**3


def measure_only(words, work_dir, paths=None, workers=None, settings=None, vector=None):
    paths = paths or default_paths()
    workers = default_workers() if workers is None else workers
    settings = settings or CountSettings()
    vector = vector or VectorSettings()
    uk, fine = text_files(paths)
    check_files(paths)
    jobs = group_order(list_groups(uk), list_groups(fine))
    overlap, skip_urls = load_overlap(work_dir, uk, fine, workers)
    runner = Runner(jobs, paths, work_dir, workers, skip_urls)
    marks = [time.monotonic()]
    counts, used, total = runner.vocabulary(settings.cleaning, words)
    marks.append(time.monotonic())
    rows = select_rows(counts)
    table = count_table(
        used,
        rows,
        settings,
        workers,
        [],
        runner.sweep_dir,
        None,
        skip_urls,
        counts,
    )
    marks.append(time.monotonic())
    updates = int(table.sum()) / ((settings.window + 1) / 2)
    matrix = ppmi_matrix(table, vector.smoothing, vector.shift)
    del table
    shutil.rmtree(runner.sweep_dir, ignore_errors=True)
    marks.append(time.monotonic())
    result, _, _ = runner.score(rows, matrix, vector)
    marks.append(time.monotonic())
    count_seconds = max(marks[2] - marks[1], 1e-9)
    print(f"text overlap: {overlap['shared_token_share']:.4f} of FineWeb tokens")
    print(f"words: {total:,} in {len(used)} of {len(jobs)} groups, rows {len(rows):,}")
    print(
        f"vocabulary {marks[1] - marks[0]:.1f}s, {total / max(marks[1] - marks[0], 1e-9):,.0f} words/s"
    )
    print(
        f"count {count_seconds:.1f}s, {total / count_seconds:,.0f} words/s, {updates / count_seconds:,.0f} pair updates/s"
    )
    print(f"ppmi {marks[3] - marks[2]:.1f}s, nonzero cells {matrix.nnz:,}")
    print(f"svd and scoring {marks[4] - marks[3]:.1f}s")
    print(f"peak RAM {peak_ram_gb():.1f} GB in the main process")
    print(f"tune score {result['tune']:.3f}, dev {result['dev']:.3f}")
    print(f"by part {result['by_part']}")
    print(f"by type {result['by_type']}")
    print(f"probes {result['probes']}")
    record_time(work_dir, marks[4] - marks[0])
    return result
