import datetime
import hashlib
import math
import os
from typing import NamedTuple

import numpy as np

import fileio
from model import evaluate, load_model

HERE = os.path.dirname(os.path.abspath(__file__))
PROBES = os.path.join(HERE, "data", "probes.txt")
FIXED_LINES = [
    "This test uses about 450 words of 28,338 at Zipf 3.",
    "Pass rule: pooled, nouns are 80% of the test.",
    "Gaps in this test:",
    "  Modern words: only electronics, dish, and shop add them. The modern tier is the only view of email, blog, and similar words.",
    "  UK/US vocabulary: voc.tab has about 100 usable rows at most, so coverage is partial.",
    "  Verbs: the test holds only 45 test verbs.",
    "  Bias: Battig is 1969 US data, and Zipf 3 excludes some UK words.",
]


class TaggedPuzzle(NamedTuple):
    words: list
    answer: str
    tags: dict


class TestRefused(Exception):
    pass


def load_tagged_puzzles(path):
    puzzles = []
    for line in fileio.read_lines(path):
        fields = line.split(";")
        tags = dict(tag.split("=", 1) for tag in fields[2].split())
        puzzles.append(TaggedPuzzle(fields[0].split(), fields[1].strip(), tags))
    return puzzles


def load_probes(path):
    probes = {}
    tier = None
    for line in fileio.read_lines(path):
        if line.startswith("# tier "):
            tier = line.removeprefix("# tier ").strip()
            probes[tier] = []
        elif line.strip():
            words, answer = line.split(";")
            probes[tier].append((words.split(), answer.strip()))
    return probes


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    middle = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    bottom = 1 + z * z / n
    return (middle - spread) / bottom, (middle + spread) / bottom


def class_bootstrap(results, targets, n_boot=2000, seed=780):
    if not results:
        return 0.0, 0.0
    names = sorted(set(targets))
    place = {name: i for i, name in enumerate(names)}
    right = np.zeros(len(names))
    total = np.zeros(len(names))
    for result, target in zip(results, targets):
        total[place[target]] += 1
        right[place[target]] += bool(result)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(names), (n_boot, len(names)))
    scores = right[picks].sum(axis=1) / total[picks].sum(axis=1)
    low, high = np.percentile(scores, [2.5, 97.5])
    return float(low), float(high)


def group(results, targets):
    k = sum(results)
    n = len(results)
    return {
        "score": k / n if n else 0.0,
        "n": n,
        "binomial": wilson(k, n),
        "bootstrap": class_bootstrap(results, targets),
    }


def group_by(puzzles, results, targets, tag):
    names = sorted({puzzle.tags[tag] for puzzle in puzzles})
    report = {}
    for name in names:
        chosen = [i for i, puzzle in enumerate(puzzles) if puzzle.tags[tag] == name]
        report[name] = group([results[i] for i in chosen], [targets[i] for i in chosen])
    return report


def puzzle_results(puzzles, model):
    return [evaluate([(p.words, p.answer)], model) == 1.0 for p in puzzles]


def score_report(puzzles, model):
    results = puzzle_results(puzzles, model)
    targets = [p.tags["target"] for p in puzzles]
    forms = group_by(puzzles, results, targets, "form")
    empty = group([], [])
    missing = [
        i
        for i, p in enumerate(puzzles)
        if any(word not in model.index for word in p.words)
    ]
    unknown = group([results[i] for i in missing], [targets[i] for i in missing])
    unknown["words"] = sorted(
        {w for i in missing for w in puzzles[i].words if w not in model.index}
    )
    return {
        "overall": group(results, targets),
        "by_part": group_by(puzzles, results, targets, "part"),
        "by_type": group_by(puzzles, results, targets, "type"),
        "by_domain": group_by(puzzles, results, targets, "domain"),
        "twins": {
            "us": forms.get("us", empty),
            "uk": forms.get("uk", empty),
            "n_pairs": len({p.tags["pair"] for p in puzzles} - {"-"}),
        },
        "unknown": unknown,
        "distinct_words": len({word for p in puzzles for word in p.words}),
    }


def format_group(name, item):
    low, high = item["binomial"]
    boot_low, boot_high = item["bootstrap"]
    return (
        f"  {name:24} {item['score']:.3f}  n={item['n']:<5} "
        f"binomial {low:.3f}-{high:.3f}  class bootstrap {boot_low:.3f}-{boot_high:.3f}"
    )


def format_report(report):
    overall = report["overall"]
    verdict = "PASS" if overall["score"] >= 0.90 else "MISS"
    lines = [f"Pooled score {overall['score']:.3f} against 0.90: {verdict}"]
    lines.append(format_group("overall", overall))
    for title, key in (
        ("By part", "by_part"),
        ("By type", "by_type"),
        ("By domain", "by_domain"),
    ):
        lines.append(f"{title}:")
        for name, item in report[key].items():
            lines.append(format_group(name, item))
    twins = report["twins"]
    lines.append(f"UK/US twins: {twins['n_pairs']} pairs")
    lines.append(format_group("us form", twins["us"]))
    lines.append(format_group("uk form", twins["uk"]))
    unknown = report["unknown"]
    lines.append(f"Puzzles with an unknown word: {unknown['n']}")
    if unknown["words"]:
        lines.append("  " + " ".join(unknown["words"]))
    if "probes" in report:
        lines.append("Probe tiers (hand-written diagnostics, never choose a setting):")
        for tier, item in report["probes"].items():
            lines.append(f"  {tier:24} {item['score']:.3f}  n={item['n']}")
    lines.append(f"Distinct words in this test: {report['distinct_words']}")
    lines.extend(FIXED_LINES)
    return "\n".join(lines)


def model_hash(folder):
    digest = hashlib.sha256()
    for name in ("vectors.npy", "words.txt", "hub.npy"):
        path = os.path.join(folder, name)
        if name == "hub.npy" and not os.path.exists(path):
            continue
        with open(path, "rb") as f:
            while block := f.read(1024 * 1024):
                digest.update(block)
    return digest.hexdigest()


def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def probe_scores(model, path=PROBES):
    scores = {}
    for tier, items in load_probes(path).items():
        scores[tier] = {"score": evaluate(items, model), "n": len(items)}
    return scores


def run_test(model_folder, puzzle_path, log_path, owner_approved=False):
    model = load_model(model_folder)
    if model.meta.get("frozen") is not True:
        raise TestRefused("The model is not frozen, so the test does not run.")
    if model.meta.get("stalled") and not owner_approved:
        raise TestRefused(
            "Tuning stalled below 0.90, so the owner must choose. "
            "Use --owner-approved only after that choice."
        )
    puzzles_sha = file_hash(puzzle_path)
    model_sha = model_hash(model_folder)
    if os.path.exists(log_path):
        for line in fileio.read_lines(log_path):
            if f"puzzles={puzzles_sha}" in line and f"model={model_sha}" not in line:
                raise TestRefused(
                    "A different model already saw this test. "
                    "Build a fresh test with a new seed, and report that earlier words were seen."
                )
    report = score_report(load_tagged_puzzles(puzzle_path), model)
    report["probes"] = probe_scores(model)
    today = datetime.date.today().isoformat()
    fileio.append_line(
        log_path,
        f"{today} puzzles={puzzles_sha} model={model_sha} "
        f"score={report['overall']['score']:.4f}",
    )
    return report
