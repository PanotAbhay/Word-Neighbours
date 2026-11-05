import collections
import dataclasses
import hashlib
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from typing import NamedTuple

import numpy as np

import fileio
from textread import Cleaning, list_groups, read_group, tokenize_text

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MAX_UINT64_COLUMNS = 10000
MAX_TABLE_COLUMNS = 20000

_worker = {}


@dataclasses.dataclass(frozen=True)
class CountSettings:
    window: int = 5
    n_columns: int = 10000
    directional: bool = False
    stoplist: str = "base30"
    subsample: float | None = None
    cleaning: Cleaning = Cleaning()

    def key(self):
        text = json.dumps(dataclasses.asdict(self), sort_keys=True)
        return hashlib.sha1(text.encode()).hexdigest()[:12]

    def table_columns(self):
        return self.n_columns * (2 if self.directional else 1)


class VocabResult(NamedTuple):
    counts: dict
    jobs_used: list
    total_words: int


def load_stoplist(name, ranked_words):
    if name == "top100":
        return set(ranked_words[:100])
    file = {"base30": "stopwords_base.txt", "full": "stopwords_extended.txt"}[name]
    return {w.strip() for w in fileio.read_lines(os.path.join(DATA, file)) if w.strip()}


def select_rows(counts, n=100000):
    return sorted(counts, key=lambda word: (-counts[word], word))[:n]


def select_columns(counts, stopwords, n):
    kept = [word for word in counts if word not in stopwords]
    return sorted(kept, key=lambda word: (-counts[word], word))[:n]


def encode_tokens(lines, index):
    ids = []
    for line in lines:
        ids.extend([index.get(token, -2) for token in line])
        ids.append(-1)
    return np.array(ids, dtype=np.int32)


def add_counts(table, seq, is_stop, col_of, window, directional):
    n_cols = table.shape[1]
    n_base = n_cols // 2 if directional else n_cols
    flat = table.reshape(-1)
    is_word = seq >= 0
    stop_token = np.zeros(len(seq), dtype=bool)
    stop_token[is_word] = is_stop[seq[is_word]]
    filtered = seq[~stop_token]
    for stream, target_mask in ((seq, is_stop), (filtered, ~is_stop)):
        sid = np.cumsum(stream == -1)
        for d in range(1, window + 1):
            if len(stream) <= d:
                break
            a, b, same = stream[:-d], stream[d:], sid[:-d] == sid[d:]
            for t, c, right in ((a, b, True), (b, a, False)):
                ok = same & (t >= 0) & (c >= 0)
                ok[ok] = target_mask[t[ok]] & (col_of[c[ok]] >= 0)
                cols = col_of[c[ok]].astype(np.int64)
                if directional and right:
                    cols += n_base
                cells = t[ok].astype(np.int64) * n_cols + cols
                np.add.at(flat, cells, flat.dtype.type(window - d + 1))


def url_hash(url):
    digest = hashlib.blake2b((url or "").encode(), digest_size=8).digest()
    return int.from_bytes(digest)


def read_rows(path, group, skip_urls):
    if skip_urls is None or path not in skip_urls["paths"]:
        return read_group(path, group, ("text",))["text"]
    data = read_group(path, group, ("text", "url"))
    hashes = skip_urls["hashes"]
    return [
        text
        for text, url in zip(data["text"], data["url"])
        if url_hash(url) not in hashes
    ]


def run_jobs(jobs, work, init, initargs, workers):
    if workers == 0:
        init(*initargs)
        for job in jobs:
            yield work(job)
        return
    context = multiprocessing.get_context("spawn")
    pool = ProcessPoolExecutor(workers, context, init, initargs)
    try:
        waiting = iter(jobs)
        pending = collections.deque()
        while True:
            while len(pending) < 2 * workers:
                job = next(waiting, None)
                if job is None:
                    break
                pending.append(pool.submit(work, job))
            if not pending:
                break
            yield pending.popleft().result()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def _init_vocab(cleaning, skip_urls):
    _worker["cleaning"] = cleaning
    _worker["skip_urls"] = skip_urls


def _vocab_job(job):
    path, group = job
    counts = collections.Counter()
    n_words = 0
    limit = _worker["cleaning"].max_word_len
    for text in read_rows(path, group, _worker["skip_urls"]):
        for line in tokenize_text(text, _worker["cleaning"]):
            n_words += len(line)
            counts.update(token for token in line if len(token) <= limit)
    return counts, n_words


def vocabulary_pass(
    jobs,
    cleaning,
    workers,
    floor=5,
    max_words=None,
    skip_urls=None,
    max_entries=20000000,
):
    counts = collections.Counter()
    used = []
    total = 0
    results = run_jobs(jobs, _vocab_job, _init_vocab, (cleaning, skip_urls), workers)
    for job, (group_counts, n_words) in zip(jobs, results):
        counts.update(group_counts)
        used.append(job)
        total += n_words
        if len(counts) > max_entries:
            counts = collections.Counter(
                {word: n for word, n in counts.items() if n >= floor}
            )
        if max_words is not None and total >= max_words:
            break
    results.close()
    kept = {word: n for word, n in counts.items() if n >= floor}
    return VocabResult(kept, used, total)


def words_hash(words):
    return hashlib.sha1("\n".join(words).encode()).hexdigest()[:12]


def jobs_hash(jobs, skip_urls):
    skipped = 0 if skip_urls is None else len(skip_urls["hashes"])
    text = json.dumps([[os.path.basename(p), g] for p, g in jobs] + [skipped])
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def checkpoint_name(checkpoint_dir, settings, hash_of_words, milestone):
    name = f"{settings.key()}_{hash_of_words}_{milestone}"
    return os.path.join(checkpoint_dir, name)


def save_checkpoint(base, table, state):
    os.makedirs(os.path.dirname(base), exist_ok=True)
    if os.path.exists(base + ".json"):
        os.remove(base + ".json")
    temp = base + ".npy.tmp"
    with open(temp, "wb") as f:
        np.save(f, table)
    os.replace(temp, base + ".npy")
    temp = base + ".json.tmp"
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(state, f)
    os.replace(temp, base + ".json")


def read_state(base):
    if not (os.path.exists(base + ".npy") and os.path.exists(base + ".json")):
        return None
    with open(base + ".json", encoding="utf-8") as f:
        return json.load(f)


def load_checkpoint(checkpoint_dir, settings, hash_of_words, milestone):
    base = checkpoint_name(checkpoint_dir, settings, hash_of_words, milestone)
    if read_state(base) is None:
        return None
    return np.load(base + ".npy")


def _init_count(words, cleaning, keep, skip_urls):
    _worker["index"] = {word: i for i, word in enumerate(words)}
    _worker["cleaning"] = cleaning
    _worker["keep"] = keep
    _worker["skip_urls"] = skip_urls


def _count_job(job):
    position, path, group = job
    ids = []
    n_words = 0
    for text in read_rows(path, group, _worker["skip_urls"]):
        lines = tokenize_text(text, _worker["cleaning"])
        n_words += sum(len(line) for line in lines)
        if lines:
            ids.append(encode_tokens(lines, _worker["index"]))
    seq = np.concatenate(ids) if ids else np.empty(0, dtype=np.int32)
    keep = _worker["keep"]
    if keep is not None and len(seq):
        rng = np.random.default_rng([780, position])
        is_word = seq >= 0
        chance = np.ones(len(seq))
        chance[is_word] = keep[seq[is_word]]
        seq = seq[rng.random(len(seq)) < chance]
    return seq, n_words


def keep_chances(words, counts, subsample):
    if subsample is None:
        return None
    frequency = np.array([counts.get(word, 0) for word in words], dtype=np.float64)
    frequency /= max(frequency.sum(), 1.0)
    frequency[frequency == 0] = 1.0
    return np.minimum(1.0, np.sqrt(subsample / frequency))


def word_masks(words, settings):
    stopwords = load_stoplist(settings.stoplist, words)
    index = {word: i for i, word in enumerate(words)}
    is_stop = np.array([word in stopwords for word in words], dtype=bool)
    columns = [word for word in words if word not in stopwords][: settings.n_columns]
    col_of = np.full(len(words), -1, dtype=np.int32)
    for j, word in enumerate(columns):
        col_of[index[word]] = j
    return is_stop, col_of


def cell_bound(words, col_of, window, counts):
    reach = sum(window - d + 1 for d in range(1, window + 1))
    largest = max(
        (counts.get(word, 0) for word, col in zip(words, col_of) if col >= 0),
        default=0,
    )
    return 2 * reach * largest


def table_dtype(words, col_of, settings, counts):
    if settings.table_columns() <= MAX_UINT64_COLUMNS:
        return np.uint64
    if counts is None:
        raise ValueError("a table over 10000 columns needs the word counts")
    if cell_bound(words, col_of, settings.window, counts) >= 2**32:
        raise ValueError("a cell of this table could pass 32 bits")
    return np.uint32


def count_table(
    jobs,
    words,
    settings,
    workers,
    milestones,
    checkpoint_dir,
    on_milestone,
    skip_urls=None,
    counts=None,
):
    if settings.table_columns() > MAX_TABLE_COLUMNS:
        raise ValueError(f"over {MAX_TABLE_COLUMNS} table columns")
    if settings.subsample is not None and counts is None:
        raise ValueError("subsampling needs the word counts")
    hash_of_words = words_hash(words)
    this_run = jobs_hash(jobs, skip_urls)
    final = checkpoint_name(checkpoint_dir, settings, hash_of_words, "final")
    state = read_state(final)
    if state is not None and state["jobs"] == this_run:
        return np.load(final + ".npy")
    is_stop, col_of = word_masks(words, settings)
    table_shape = (len(words), settings.table_columns())
    table, done, seen, resumed = None, 0, 0, None
    waiting = sorted(milestones)
    for milestone in reversed(waiting):
        base = checkpoint_name(checkpoint_dir, settings, hash_of_words, milestone)
        state = read_state(base)
        if state is not None and state["jobs"] == this_run:
            table = np.load(base + ".npy")
            done, seen, resumed = state["done"], state["words"], milestone
            break
    if table is None:
        dtype = table_dtype(words, col_of, settings, counts)
        table = np.zeros(table_shape, dtype=dtype)
    waiting = [m for m in waiting if m > seen]
    if resumed is not None and on_milestone is not None:
        on_milestone(resumed, seen, table)
    keep = keep_chances(words, counts, settings.subsample)
    work = [(i, path, group) for i, (path, group) in enumerate(jobs)][done:]
    results = run_jobs(
        work,
        _count_job,
        _init_count,
        (words, settings.cleaning, keep, skip_urls),
        workers,
    )
    for job, (seq, n_words) in zip(work, results):
        add_counts(table, seq, is_stop, col_of, settings.window, settings.directional)
        seen += n_words
        done = job[0] + 1
        while waiting and seen >= waiting[0]:
            base = checkpoint_name(checkpoint_dir, settings, hash_of_words, waiting[0])
            save_checkpoint(
                base, table, {"jobs": this_run, "done": done, "words": seen}
            )
            milestone = waiting.pop(0)
            if on_milestone is not None:
                on_milestone(milestone, seen, table)
    save_checkpoint(final, table, {"jobs": this_run, "done": done, "words": seen})
    return table


def _init_overlap():
    pass


def _overlap_job(job):
    data = read_group(job[0], job[1], ("url", "token_count"))
    return [url_hash(url) for url in data["url"]], data["token_count"]


def measure_overlap(uk_paths, fine_paths, workers=0):
    uk_hashes = set()
    for hashes, _ in run_jobs(
        list_groups(uk_paths), _overlap_job, _init_overlap, (), workers
    ):
        uk_hashes.update(hashes)
    fine_rows = shared_rows = fine_tokens = shared_tokens = 0
    shared = set()
    for hashes, tokens in run_jobs(
        list_groups(fine_paths), _overlap_job, _init_overlap, (), workers
    ):
        for value, n_tokens in zip(hashes, tokens):
            fine_rows += 1
            fine_tokens += n_tokens or 0
            if value in uk_hashes:
                shared_rows += 1
                shared_tokens += n_tokens or 0
                shared.add(value)
    return {
        "uk_rows": len(uk_hashes),
        "fine_rows": fine_rows,
        "shared_rows": shared_rows,
        "shared_token_share": shared_tokens / fine_tokens if fine_tokens else 0.0,
        "skip_urls": {"hashes": shared, "paths": set(fine_paths)},
    }
