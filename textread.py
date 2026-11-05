import dataclasses
import random
import re
import unicodedata

import fileio

WORD = re.compile("[a-z]+")
SENTENCE_BREAK = re.compile(r"[.!?\n]+")
APOSTROPHE = re.compile("(?<=[a-z])['’](?=[a-z])")

_open_files = {}


@dataclasses.dataclass(frozen=True)
class Cleaning:
    breaks: str = "line"
    apostrophes: str = "split"
    accents: str = "split"
    min_line_words: int = 5
    max_word_len: int = 25


def fold_accents(text):
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def tokenize_text(text, cleaning):
    if not text:
        return []
    text = text.lower()
    if cleaning.apostrophes == "join":
        text = APOSTROPHE.sub("", text)
    if cleaning.accents == "fold":
        text = fold_accents(text)
    if cleaning.breaks == "sentence":
        parts = SENTENCE_BREAK.split(text)
    else:
        parts = text.split("\n")
    lines = []
    for part in parts:
        tokens = WORD.findall(part)
        if tokens and len(tokens) >= cleaning.min_line_words:
            lines.append(tokens)
    return lines


def parquet_file(path):
    import pyarrow.parquet as pq

    if path not in _open_files:
        _open_files[path] = pq.ParquetFile(path)
    return _open_files[path]


def list_groups(paths):
    groups = []
    for path in paths:
        for group in range(parquet_file(path).num_row_groups):
            groups.append((path, group))
    return groups


def read_group(path, group, columns):
    table = parquet_file(path).read_row_group(group, columns=list(columns))
    return {name: table.column(name).to_pylist() for name in columns}


def group_order(uk, fine, seed=780):
    rng = random.Random(seed)
    uk = list(uk)
    fine = list(fine)
    rng.shuffle(uk)
    rng.shuffle(fine)
    order = []
    used = 0
    for i, group in enumerate(uk):
        upto = (i + 1) * len(fine) // len(uk)
        order.extend(fine[used:upto])
        used = upto
        order.append(group)
    order.extend(fine[used:])
    return order


def read_text_file(path):
    return " ".join(fileio.read_text(path).split("\n")).replace("\r", "")
