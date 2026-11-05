import gc
import os
import tempfile
import unittest

import numpy as np

import vectormath
from model import (
    Model,
    ModelError,
    evaluate,
    hub_scores,
    load_model,
    most_similar,
    odd_one_out,
    save_model,
)
from scoring import model_hash

WORDS = ["apple", "pear", "plum", "chair", "sofa"]
VECTORS = np.array(
    [
        [1.0, 0.1, 0.0],
        [0.9, 0.2, 0.0],
        [0.8, 0.3, 0.1],
        [0.0, 0.1, 1.0],
        [0.1, 0.0, 0.9],
    ],
    dtype=np.float32,
)
VECTORS /= np.linalg.norm(VECTORS, axis=1, keepdims=True)
RULE_WORDS = ["a", "b", "c", "d"]
RULE_VECTORS = np.array(
    [
        [0.7, -0.7, 0.6],
        [0.8, 0.5, -0.7],
        [0.1, 0.9, 0.2],
        [-0.3, -0.5, -0.7],
    ],
    dtype=np.float32,
)


def rule_model(meta):
    index = {word: i for i, word in enumerate(RULE_WORDS)}
    return Model(RULE_WORDS, index, RULE_VECTORS, meta)


class ModelTest(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(folder.cleanup)
        self.folder = os.path.join(folder.name, "model")
        save_model(self.folder, WORDS, VECTORS, {"k": 1})
        self.model = load_model(self.folder)

    def test_round_trip_float32_and_mmap(self):
        self.assertEqual(self.model.words, WORDS)
        self.assertEqual(self.model.index["plum"], 2)
        self.assertEqual(self.model.vectors.dtype, np.float32)
        self.assertIsInstance(self.model.vectors, np.memmap)
        self.assertEqual(self.model.meta["k"], 1)

    def test_hub_scores_mean_of_nearest_without_self(self):
        scores = hub_scores(VECTORS, k=2, chunk_rows=2)
        cosines = VECTORS @ VECTORS.T
        np.fill_diagonal(cosines, -np.inf)
        expected = np.sort(cosines, axis=1)[:, -2:].mean(axis=1)
        self.assertTrue(np.allclose(scores, expected))

    def test_hub_word_stops_pulling_towards_every_word(self):
        gram = np.array(
            [
                [1.0, 0.5, 0.5, 0.6],
                [0.5, 1.0, 0.5, 0.6],
                [0.5, 0.5, 1.0, 0.6],
                [0.6, 0.6, 0.6, 1.0],
            ]
        )
        words = ["x", "y", "z", "hub"]
        index = {word: i for i, word in enumerate(words)}
        vectors = np.linalg.cholesky(gram)
        plain = Model(words, index, vectors, {})
        self.assertEqual(odd_one_out(words, plain), "x")
        corrected = plain._replace(hub=np.array([0.5, 0.5, 0.5, 0.9]))
        self.assertEqual(odd_one_out(words, corrected), "hub")

    def test_hub_round_trip_and_removal(self):
        self.assertIsNone(self.model.hub)
        hub = np.arange(5, dtype=np.float32)
        other = self.folder + "_hub"
        save_model(other, WORDS, VECTORS, {}, hub=hub)
        self.assertTrue(np.array_equal(load_model(other).hub, hub))
        gc.collect()
        save_model(other, WORDS, VECTORS, {})
        self.assertIsNone(load_model(other).hub)

    def test_hub_of_wrong_length_is_model_error(self):
        other = self.folder + "_hub"
        save_model(other, WORDS, VECTORS, {}, hub=np.zeros(4))
        with self.assertRaises(ModelError):
            load_model(other)

    def test_model_hash_covers_the_hub(self):
        other = self.folder + "_hub"
        save_model(other, WORDS, VECTORS, {"k": 1}, hub=np.zeros(5))
        self.assertNotEqual(model_hash(other), model_hash(self.folder))

    def test_odd_one_out(self):
        puzzle = ["apple", "pear", "plum", "chair"]
        self.assertEqual(odd_one_out(puzzle, self.model), "chair")

    def test_matches_vectormath_cosine(self):
        puzzle = ["apple", "pear", "sofa", "chair"]
        totals = []
        for i, word in enumerate(puzzle):
            vector = VECTORS[WORDS.index(word)].tolist()
            totals.append(
                sum(
                    vectormath.cosine(vector, VECTORS[WORDS.index(other)].tolist())
                    for j, other in enumerate(puzzle)
                    if i != j
                )
            )
        want = puzzle[int(np.argmin(totals))]
        self.assertEqual(odd_one_out(puzzle, self.model), want)

    def test_zero_vector_counts_as_zero_cosine(self):
        vectors = VECTORS.copy()
        vectors[3] = 0.0
        other = self.folder + "_zero"
        save_model(other, WORDS, vectors, {})
        model = load_model(other)
        puzzle = ["apple", "pear", "plum", "chair"]
        self.assertEqual(odd_one_out(puzzle, model), "chair")

    def test_most_similar_excludes_self_and_sorts(self):
        got = most_similar("apple", self.model, 2)
        self.assertEqual(got[0][0], "pear")
        self.assertEqual(len(got), 2)
        self.assertGreater(got[0][1], got[1][1])
        self.assertNotIn("apple", [word for word, _ in got])

    def test_most_similar_unknown_is_empty(self):
        self.assertEqual(most_similar("zzz", self.model, 2), [])

    def test_evaluate_unknown_word_is_wrong(self):
        puzzles = [(["apple", "pear", "plum", "zzz"], "zzz")]
        self.assertEqual(evaluate(puzzles, self.model), 0.0)

    def test_evaluate_scores_right_answers(self):
        puzzles = [
            (["apple", "pear", "plum", "chair"], "chair"),
            (["apple", "pear", "plum", "chair"], "apple"),
        ]
        self.assertEqual(evaluate(puzzles, self.model), 0.5)

    def test_missing_folder_raises_model_error(self):
        with self.assertRaises(ModelError):
            load_model(os.path.join(self.folder, "nothing"))

    def test_length_mismatch_raises_model_error(self):
        with open(os.path.join(self.folder, "words.txt"), "a", encoding="utf-8") as f:
            f.write("extra\n")
        with self.assertRaises(ModelError):
            load_model(self.folder)

    def test_truncated_npy_raises_model_error(self):
        path = os.path.join(self.folder, "vectors.npy")
        del self.model
        with open(path, "rb") as f:
            data = f.read()
        with open(path, "wb") as f:
            f.write(data[: len(data) // 2])
        with self.assertRaises(ModelError):
            load_model(self.folder)


class RuleTest(unittest.TestCase):
    def test_sum_is_the_default_rule(self):
        self.assertEqual(odd_one_out(RULE_WORDS, rule_model({})), "d")
        self.assertEqual(odd_one_out(RULE_WORDS, rule_model({"rule": "sum"})), "d")

    def test_median_rule_picks_the_lowest_median(self):
        self.assertEqual(odd_one_out(RULE_WORDS, rule_model({"rule": "median"})), "c")

    def test_max_rule_picks_the_lowest_best_match(self):
        self.assertEqual(odd_one_out(RULE_WORDS, rule_model({"rule": "max"})), "a")

    def test_rules_apply_after_the_hub_correction(self):
        hub = np.array([0.0, 0.0, 0.0, 5.0])
        model = rule_model({"rule": "max"})._replace(hub=hub)
        self.assertEqual(odd_one_out(RULE_WORDS, model), "d")

    def test_unknown_rule_is_model_error(self):
        other = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(other.cleanup)
        folder = os.path.join(other.name, "model")
        save_model(folder, RULE_WORDS, RULE_VECTORS, {"rule": "mean"})
        with self.assertRaises(ModelError):
            load_model(folder)


class HalfPrecisionTest(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(folder.cleanup)
        self.folder = os.path.join(folder.name, "model")
        save_model(self.folder, WORDS, VECTORS, {"bits": 16})
        self.model = load_model(self.folder)
        self.stored = np.asarray(self.model.vectors, dtype=np.float32)

    def test_bits_16_stores_float16(self):
        self.assertEqual(self.model.vectors.dtype, np.float16)
        self.assertEqual(self.model.vectors.shape, VECTORS.shape)
        self.assertTrue(np.allclose(self.stored, VECTORS, atol=1e-3))

    def test_other_bit_sizes_are_refused(self):
        with self.assertRaises(ValueError):
            save_model(self.folder + "_8", WORDS, VECTORS, {"bits": 8})

    def test_odd_one_out_on_half_precision(self):
        puzzle = ["apple", "pear", "plum", "chair"]
        self.assertEqual(odd_one_out(puzzle, self.model), "chair")

    def test_most_similar_multiplies_in_float32(self):
        got = dict(most_similar("apple", self.model, 4))
        wide = self.stored @ self.stored[0]
        for word, score in got.items():
            self.assertAlmostEqual(score, float(wide[WORDS.index(word)]), places=6)

    def test_most_similar_chunks_do_not_change_result(self):
        one = most_similar("apple", self.model, 4, chunk_rows=2)
        two = most_similar("apple", self.model, 4)
        self.assertEqual([w for w, _ in one], [w for w, _ in two])
        self.assertTrue(np.allclose([s for _, s in one], [s for _, s in two]))

    def test_hub_scores_multiply_in_float32(self):
        got = hub_scores(self.model.vectors, k=2, chunk_rows=2)
        want = hub_scores(self.stored, k=2, chunk_rows=2)
        self.assertTrue(np.allclose(got, want, rtol=1e-6, atol=1e-7))


if __name__ == "__main__":
    unittest.main()
