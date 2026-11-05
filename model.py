import json
import os
from typing import NamedTuple

import numpy as np

import fileio
import vectormath


RULES = {
    "sum": lambda off_diagonal: off_diagonal.sum(axis=1),
    "median": lambda off_diagonal: np.median(off_diagonal, axis=1),
    "max": lambda off_diagonal: off_diagonal.max(axis=1),
}
BIT_SIZES = {16: np.float16, 32: np.float32}


class ModelError(Exception):
    pass


class Model(NamedTuple):
    words: list
    index: dict
    vectors: np.ndarray
    meta: dict
    hub: np.ndarray | None = None


def hub_scores(vectors, k=10, chunk_rows=2000):
    vectors = np.asarray(vectors, dtype=np.float32)
    scores = np.zeros(len(vectors), dtype=np.float32)
    for start in range(0, len(vectors), chunk_rows):
        rows = np.arange(start, min(start + chunk_rows, len(vectors)))
        cosines = vectors[rows] @ vectors.T
        cosines[rows - start, rows] = -np.inf
        nearest = np.partition(cosines, -k, axis=1)[:, -k:]
        scores[rows] = nearest.mean(axis=1)
    return scores


def save_model(folder, words, vectors, meta, hub=None):
    bits = meta.get("bits", 32)
    if bits not in BIT_SIZES:
        raise ValueError(f"cannot store vectors in {bits} bits")
    os.makedirs(folder, exist_ok=True)
    hub_path = os.path.join(folder, "hub.npy")
    if hub is None:
        if os.path.exists(hub_path):
            os.remove(hub_path)
    else:
        temp = hub_path + ".tmp"
        with open(temp, "wb") as f:
            np.save(f, np.asarray(hub, dtype=np.float32))
        os.replace(temp, hub_path)
    temp = os.path.join(folder, "vectors.npy.tmp")
    with open(temp, "wb") as f:
        np.save(f, np.asarray(vectors, dtype=BIT_SIZES[bits]))
    os.replace(temp, os.path.join(folder, "vectors.npy"))
    temp = os.path.join(folder, "words.txt.tmp")
    fileio.write_lines(temp, words)
    os.replace(temp, os.path.join(folder, "words.txt"))
    temp = os.path.join(folder, "meta.json.tmp")
    with open(temp, "w", encoding="utf-8", newline="") as f:
        json.dump(meta, f, indent=1)
    os.replace(temp, os.path.join(folder, "meta.json"))


def load_model(folder):
    try:
        words = fileio.read_lines(os.path.join(folder, "words.txt"))
        vectors = np.load(os.path.join(folder, "vectors.npy"), mmap_mode="r")
        with open(os.path.join(folder, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        hub_path = os.path.join(folder, "hub.npy")
        hub = np.load(hub_path) if os.path.exists(hub_path) else None
    except (OSError, ValueError, EOFError) as e:
        raise ModelError(f"cannot read the model in {folder}: {e}") from e
    if vectors.ndim != 2 or len(words) != len(vectors):
        raise ModelError(f"words and vectors disagree in length in {folder}")
    if hub is not None and len(hub) != len(words):
        raise ModelError(f"words and hub scores disagree in length in {folder}")
    if meta.get("rule", "sum") not in RULES:
        raise ModelError(f"unknown odd-one-out rule {meta['rule']!r} in {folder}")
    index = {word: i for i, word in enumerate(words)}
    return Model(words, index, vectors, meta, hub)


def odd_one_out(words, model):
    ids = [model.index[word] for word in words]
    vectors = np.array(model.vectors[ids], dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)
    similarity = vectors @ vectors.T
    if model.hub is not None:
        hub = np.asarray(model.hub[ids], dtype=np.float64)
        similarity = 2 * similarity - hub[:, None] - hub[None, :]
    n = len(words)
    off_diagonal = similarity[~np.eye(n, dtype=bool)].reshape(n, n - 1)
    rule = RULES[model.meta.get("rule", "sum")]
    return words[int(np.argmin(rule(off_diagonal)))]


def most_similar(word, model, n, chunk_rows=20000):
    if word not in model.index:
        return []
    i = model.index[word]
    target = np.asarray(model.vectors[i], dtype=np.float32)
    scores = np.empty(len(model.vectors), dtype=np.float32)
    for start in range(0, len(scores), chunk_rows):
        block = np.asarray(model.vectors[start : start + chunk_rows], dtype=np.float32)
        scores[start : start + chunk_rows] = block @ target
    scores[i] = -np.inf
    n = min(n, len(scores) - 1)
    top = np.argpartition(-scores, n - 1)[:n] if n > 0 else []
    top = sorted(top, key=lambda j: (-scores[j], model.words[j]))
    return [(model.words[j], float(scores[j])) for j in top]


def evaluate(puzzles, model):
    predicted = [
        odd_one_out(words, model)
        if all(word in model.index for word in words)
        else None
        for words, _ in puzzles
    ]
    actual = [answer for _, answer in puzzles]
    return vectormath.accuracy(predicted, actual)
