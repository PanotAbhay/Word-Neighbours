import math
import unittest

import numpy as np
import scipy.sparse as sp

from vectors import (
    VectorSettings,
    embed,
    no_svd_vectors,
    ppmi_matrix,
    randomized_svd,
    unit_rows,
)

TABLE = np.array(
    [[10, 0, 3, 1], [2, 8, 0, 0], [0, 0, 0, 0], [1, 1, 6, 9], [5, 0, 0, 2]],
    dtype=np.uint64,
)


def random_matrix(rows=300, cols=100, density=0.1):
    rng = np.random.default_rng(780)
    return sp.random(rows, cols, density, format="csr", rng=rng, dtype=np.float32)


class PpmiTest(unittest.TestCase):
    def test_ppmi_matches_legacy_formula(self):
        got = ppmi_matrix(TABLE).toarray()
        row_totals = TABLE.sum(axis=1).astype(float)
        col_totals = TABLE.sum(axis=0).astype(float)
        z = sum(c**0.75 for c in col_totals)
        for w in range(TABLE.shape[0]):
            for c in range(TABLE.shape[1]):
                want = 0.0
                if TABLE[w, c]:
                    value = (
                        float(TABLE[w, c]) * z / (row_totals[w] * col_totals[c] ** 0.75)
                    )
                    want = max(0.0, math.log(value))
                self.assertAlmostEqual(float(got[w, c]), want, places=5)
        self.assertEqual(got[2].sum(), 0.0)

    def test_ppmi_is_float32_csr(self):
        got = ppmi_matrix(TABLE)
        self.assertTrue(sp.issparse(got))
        self.assertEqual(got.format, "csr")
        self.assertEqual(got.dtype, np.float32)
        self.assertTrue(np.all(got.data > 0))

    def test_ppmi_chunks_do_not_change_result(self):
        one = ppmi_matrix(TABLE, chunk_rows=2)
        two = ppmi_matrix(TABLE, chunk_rows=100)
        self.assertTrue(np.array_equal(one.toarray(), two.toarray()))
        self.assertTrue(one.has_sorted_indices)

    def test_ppmi_smoothing_changes_values(self):
        plain = ppmi_matrix(TABLE, smoothing=1.0).toarray()
        smooth = ppmi_matrix(TABLE).toarray()
        self.assertFalse(np.allclose(plain, smooth))

    def test_ppmi_keeps_table_type(self):
        table = TABLE.copy()
        ppmi_matrix(table)
        self.assertEqual(table.dtype, np.uint64)
        self.assertTrue(np.array_equal(table, TABLE))

    def test_ppmi_shift_one_changes_nothing(self):
        one = ppmi_matrix(TABLE, shift=1.0).toarray()
        self.assertTrue(np.array_equal(one, ppmi_matrix(TABLE).toarray()))

    def test_ppmi_shift_subtracts_log_shift(self):
        base = ppmi_matrix(TABLE).toarray()
        shifted = ppmi_matrix(TABLE, shift=2.0)
        want = np.maximum(base - math.log(2.0), 0.0)
        self.assertTrue(np.allclose(shifted.toarray(), want, atol=1e-5))
        self.assertLess(shifted.nnz, ppmi_matrix(TABLE).nnz)
        self.assertTrue(np.all(shifted.data > 0))

    def test_ppmi_uint32_table_gives_same_result(self):
        small = TABLE.astype(np.uint32)
        got = ppmi_matrix(small).toarray()
        self.assertTrue(np.array_equal(got, ppmi_matrix(TABLE).toarray()))

    def test_ppmi_uint32_row_total_above_32_bits(self):
        big = np.array(
            [[3_000_000_000, 3_000_000_000, 1], [1, 2, 3_000_000_000]], dtype=np.uint32
        )
        got = ppmi_matrix(big).toarray()
        want = ppmi_matrix(big.astype(np.uint64)).toarray()
        self.assertTrue(np.array_equal(got, want))
        self.assertGreater(got.sum(), 0.0)


class SvdTest(unittest.TestCase):
    def test_svd_close_to_exact(self):
        matrix = random_matrix()
        _, s = randomized_svd(matrix, 20)
        exact = np.linalg.svd(matrix.toarray(), compute_uv=False)[:20]
        self.assertEqual(len(s), 20)
        self.assertTrue(np.all(np.abs(s - exact) <= 0.01 * exact))

    def test_svd_left_vectors_reconstruct(self):
        matrix = random_matrix()
        u, s = randomized_svd(matrix, 20)
        self.assertEqual(u.shape, (300, 20))
        identity = u.T @ u
        self.assertTrue(np.allclose(identity, np.eye(20), atol=1e-4))
        projected = np.linalg.norm(u.T @ matrix.toarray(), axis=1)
        self.assertTrue(np.allclose(projected, s, rtol=1e-3))

    def test_svd_deterministic(self):
        matrix = random_matrix()
        one, s_one = randomized_svd(matrix, 20)
        two, s_two = randomized_svd(matrix, 20)
        self.assertTrue(np.array_equal(one, two))
        self.assertTrue(np.array_equal(s_one, s_two))

    def test_svd_seed_changes_result(self):
        matrix = random_matrix()
        one, _ = randomized_svd(matrix, 20)
        two, _ = randomized_svd(matrix, 20, seed=781)
        self.assertFalse(np.array_equal(one, two))


class EmbedTest(unittest.TestCase):
    def test_embed_unit_length_and_zero_rows(self):
        vectors = embed(ppmi_matrix(TABLE), VectorSettings(dims=3))
        self.assertEqual(vectors.dtype, np.float32)
        norms = np.linalg.norm(vectors, axis=1)
        self.assertEqual(norms[2], 0.0)
        for i in (0, 1, 3, 4):
            self.assertAlmostEqual(float(norms[i]), 1.0, delta=1e-5)

    def test_embed_zero_row_on_large_matrix(self):
        matrix = random_matrix().tolil()
        matrix[7] = 0
        vectors = embed(matrix.tocsr(), VectorSettings(dims=20))
        self.assertEqual(float(np.abs(vectors[7]).sum()), 0.0)

    def test_embed_clips_k_on_tiny_matrix(self):
        table = np.array(
            [[3, 1, 0, 2], [0, 4, 1, 0], [2, 0, 5, 1], [1, 1, 1, 6], [4, 0, 0, 1]],
            dtype=np.uint64,
        )
        vectors = embed(ppmi_matrix(table), VectorSettings(dims=300))
        self.assertEqual(vectors.shape[0], 5)
        self.assertEqual(vectors.shape[1], 3)

    def test_centre_flag(self):
        matrix = random_matrix()
        on = embed(matrix, VectorSettings(dims=20, centre=True))
        off = embed(matrix, VectorSettings(dims=20, centre=False))
        mean_on = np.linalg.norm(on.mean(axis=0))
        mean_off = np.linalg.norm(off.mean(axis=0))
        self.assertLess(mean_on, 0.2)
        self.assertLess(mean_on, mean_off)

    def test_p_changes_vectors(self):
        matrix = random_matrix()
        half = embed(matrix, VectorSettings(dims=20, p=0.5))
        full = embed(matrix, VectorSettings(dims=20, p=1.0))
        self.assertFalse(np.allclose(half, full))

    def test_new_settings_default_to_the_old_method(self):
        settings = VectorSettings()
        self.assertEqual(settings.shift, 1.0)
        self.assertEqual(settings.skip, 0)
        self.assertEqual(settings.rule, "sum")

    def test_default_settings_match_the_old_embed(self):
        matrix = random_matrix()
        u, s = randomized_svd(matrix, 20)
        want = unit_rows((u * s**0.5).astype(np.float32))
        want = unit_rows(want - want.mean(axis=0))
        got = embed(matrix, VectorSettings(dims=20))
        self.assertTrue(np.allclose(got, want, atol=1e-5))

    def test_skip_drops_the_leading_components(self):
        matrix = random_matrix()
        u, s = randomized_svd(matrix, 5)
        want = unit_rows((u * s**0.5).astype(np.float32)[:, 2:5])
        got = embed(matrix, VectorSettings(dims=3, skip=2, centre=False))
        self.assertEqual(got.shape, (300, 3))
        self.assertTrue(np.allclose(got, want, atol=1e-5))

    def test_skip_clips_on_tiny_matrix(self):
        vectors = embed(ppmi_matrix(TABLE), VectorSettings(dims=300, skip=1))
        self.assertEqual(vectors.shape, (5, 2))

    def test_negative_p_favours_small_components(self):
        matrix = random_matrix()
        u, s = randomized_svd(matrix, 20)
        want = unit_rows((u * s**-0.5).astype(np.float32))
        got = embed(matrix, VectorSettings(dims=20, p=-0.5, centre=False))
        self.assertTrue(np.all(np.isfinite(got)))
        self.assertTrue(np.allclose(got, want, atol=1e-5))


class NoSvdTest(unittest.TestCase):
    def test_no_svd_matches_dense_centring(self):
        matrix = ppmi_matrix(TABLE)
        dense = matrix.toarray().astype(np.float64)
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        live = norms[:, 0] > 0
        dense[live] /= norms[live]
        mean = dense[live].mean(axis=0)
        want = dense[3] - mean
        want /= np.linalg.norm(want)
        got = no_svd_vectors(matrix, [3, 0])
        self.assertEqual(got.shape, (2, TABLE.shape[1]))
        self.assertTrue(np.allclose(got[0], want, atol=1e-5))

    def test_no_svd_zero_row_stays_zero(self):
        got = no_svd_vectors(ppmi_matrix(TABLE), [2])
        self.assertEqual(float(np.abs(got).sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
