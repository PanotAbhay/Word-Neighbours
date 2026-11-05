import dataclasses

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclasses.dataclass(frozen=True)
class VectorSettings:
    dims: int = 300
    p: float = 0.5
    centre: bool = True
    smoothing: float = 0.75
    shift: float = 1.0
    skip: int = 0
    rule: str = "sum"


def ppmi_matrix(table, smoothing=0.75, shift=1.0, chunk_rows=2000):
    row_totals = table.sum(axis=1, dtype=np.uint64).astype(np.float64)
    col_power = table.sum(axis=0, dtype=np.uint64).astype(np.float64) ** smoothing
    total = col_power.sum()
    log_shift = np.log(shift)
    data, indices = [], []
    indptr = np.zeros(table.shape[0] + 1, dtype=np.int64)
    for start in range(0, table.shape[0], chunk_rows):
        chunk = table[start : start + chunk_rows]
        rows, cols = np.nonzero(chunk)
        values = chunk[rows, cols].astype(np.float64) * total
        pmi = np.log(values / (row_totals[start + rows] * col_power[cols])) - log_shift
        keep = pmi > 0
        data.append(pmi[keep].astype(np.float32))
        indices.append(cols[keep].astype(np.int32))
        per_row = np.bincount(rows[keep], minlength=chunk.shape[0])
        indptr[start + 1 : start + chunk.shape[0] + 1] = per_row
    indptr = np.cumsum(indptr)
    data = np.concatenate(data) if data else np.empty(0, dtype=np.float32)
    indices = np.concatenate(indices) if indices else np.empty(0, dtype=np.int32)
    return sp.csr_matrix((data, indices, indptr), shape=table.shape)


def clipped_k(shape, k):
    return max(1, min(k, min(shape) - 1))


def randomized_svd(matrix, k, oversample=50, n_iter=4, seed=780):
    k = clipped_k(matrix.shape, k)
    if k + oversample >= min(matrix.shape):
        u, s, _ = np.linalg.svd(matrix.toarray(), full_matrices=False)
        return u[:, :k], s[:k]
    rng = np.random.default_rng(seed)
    block = rng.standard_normal((matrix.shape[1], k + oversample), dtype=np.float32)
    q, _ = np.linalg.qr(matrix @ block)
    for _ in range(n_iter):
        z, _ = np.linalg.qr(matrix.T @ q)
        q, _ = np.linalg.qr(matrix @ z)
    small = (matrix.T @ q).T
    u, s, _ = np.linalg.svd(small, full_matrices=False)
    return q @ u[:, :k], s[:k]


def unit_rows(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def embed(matrix, settings):
    u, s = randomized_svd(matrix, settings.dims + settings.skip)
    u, s = u[:, settings.skip :], s[settings.skip :]
    vectors = (u * s**settings.p).astype(np.float32)
    live = matrix.getnnz(axis=1) > 0
    vectors[~live] = 0.0
    vectors = unit_rows(vectors)
    if settings.centre:
        vectors[live] -= vectors[live].mean(axis=0)
        vectors = unit_rows(vectors)
    return vectors


def no_svd_vectors(matrix, row_ids):
    norms = spla.norm(matrix, axis=1)
    live = norms > 0
    inverse = np.zeros(len(norms))
    inverse[live] = 1.0 / norms[live]
    mean = (matrix.T @ inverse) / max(int(live.sum()), 1)
    rows = matrix[list(row_ids)].toarray().astype(np.float64)
    is_live = live[list(row_ids)]
    rows = unit_rows(rows)
    rows[is_live] -= mean
    return unit_rows(rows).astype(np.float32)
