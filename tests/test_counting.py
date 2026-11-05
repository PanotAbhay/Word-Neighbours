import dataclasses
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import textbasics
import fileio
from counting import (
    CountSettings,
    add_counts,
    checkpoint_name,
    count_table,
    encode_tokens,
    load_checkpoint,
    load_stoplist,
    measure_overlap,
    read_state,
    save_checkpoint,
    select_columns,
    select_rows,
    vocabulary_pass,
    words_hash,
)
from tests.legacy_counter import (
    clean,
    flatten,
    old_counts,
    remove_words,
    split_sentences,
    tokenize_sentences,
)
from textread import Cleaning

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALICE = os.path.join(ROOT, "data", "sample_texts", "alice's_adventures_in_wonderland.txt")
LINE = "red green blue red green blue"
DEAD_WORKER = """
from concurrent.futures.process import BrokenProcessPool
from counting import run_jobs
from tests.test_counting import die, no_setup

try:
    list(run_jobs([1, 2, 3], die, no_setup, (), 1))
except BrokenProcessPool:
    print("stopped")
"""


def die(job):
    os._exit(1)


def no_setup():
    pass


def count_sentences(
    sentences, words, columns, stopwords, window, directional=False, dtype=np.uint64
):
    index = {word: i for i, word in enumerate(words)}
    is_stop = np.array([word in stopwords for word in words], dtype=bool)
    col_of = np.full(len(words), -1, dtype=np.int32)
    for j, word in enumerate(columns):
        col_of[index[word]] = j
    n_cols = len(columns) * (2 if directional else 1)
    table = np.zeros((len(words), n_cols), dtype=dtype)
    seq = encode_tokens(sentences, index)
    add_counts(table, seq, is_stop, col_of, window, directional)
    return table


class AddCountsTest(unittest.TestCase):
    def test_matches_legacy_on_alice(self):
        sentences = tokenize_sentences(
            [clean(s) for s in split_sentences(fileio.read_text(ALICE))]
        )
        stopwords = load_stoplist("base30", [])
        cooc, ctx, words = old_counts(sentences, stopwords, 2, 1200)
        counts = textbasics.count_words(
            flatten([remove_words(s, stopwords) for s in sentences])
        )
        self.assertEqual(select_columns(counts, stopwords, 1200), ctx)
        table = count_sentences(sentences, words, ctx, stopwords, 2)
        row = {word: i for i, word in enumerate(words)}
        col = {word: j for j, word in enumerate(ctx)}
        want = {pair: v for pair, v in cooc.items() if pair[1] in col}
        for (w, c), v in want.items():
            self.assertEqual(int(table[row[w], col[c]]), v)
        self.assertEqual(int(table.sum()), sum(want.values()))

    def test_no_pairs_across_breaks(self):
        words = list("abcd")
        table = count_sentences([["a", "b"], ["c", "d"]], words, words, set(), 2)
        self.assertEqual(int(table[1, 2]), 0)
        self.assertEqual(int(table[0, 1]), 2)

    def test_unknown_word_keeps_distance(self):
        words = ["a", "b"]
        for window, want in ((1, 0), (2, 1)):
            table = count_sentences([["a", "zzz", "b"]], words, words, set(), window)
            self.assertEqual(int(table[0, 1]), want)

    def test_weight_is_k_minus_d_plus_1(self):
        table = count_sentences([["a", "b"]], ["a", "b"], ["a", "b"], set(), 5)
        self.assertEqual(int(table[0, 1]), 5)

    def test_stop_targets_use_unfiltered_text(self):
        words = ["the", "a", "b"]
        table = count_sentences(
            [["the", "a", "the", "b"]], words, ["a", "b"], {"the"}, 1
        )
        self.assertEqual(int(table[0, 0]), 2)
        self.assertEqual(int(table[0, 1]), 1)
        self.assertEqual(int(table[1, 1]), 1)

    def test_stop_words_get_no_columns(self):
        counts = {"the": 10, "a": 5, "b": 7, "c": 7}
        self.assertEqual(select_columns(counts, {"the"}, 2), ["b", "c"])

    def test_directional_columns(self):
        sentences = [["a", "b", "c"], ["c", "a"]]
        words = ["a", "b", "c"]
        plain = count_sentences(sentences, words, words, set(), 2)
        both = count_sentences(sentences, words, words, set(), 2, directional=True)
        self.assertEqual(both.shape, (3, 6))
        self.assertTrue(np.array_equal(both[:, :3] + both[:, 3:], plain))
        self.assertEqual(int(both[0, 1]), 0)
        self.assertEqual(int(both[0, 4]), 2)

    def test_table_is_uint64(self):
        table = count_sentences([["a", "b"]], ["a", "b"], ["a", "b"], set(), 1)
        self.assertEqual(table.dtype, np.uint64)

    def test_uint32_table_gives_the_same_counts(self):
        sentences = [["a", "b", "c"], ["c", "a", "b", "a"]]
        words = ["a", "b", "c"]
        wide = count_sentences(sentences, words, words, set(), 3)
        small = count_sentences(sentences, words, words, set(), 3, dtype=np.uint32)
        self.assertEqual(small.dtype, np.uint32)
        self.assertTrue(np.array_equal(small, wide))
        self.assertGreater(int(small.sum()), 0)

    def test_select_rows_order(self):
        counts = {"b": 2, "a": 2, "c": 3}
        self.assertEqual(select_rows(counts, 2), ["c", "a"])


class SettingsTest(unittest.TestCase):
    def test_settings_key_changes(self):
        base = CountSettings()
        changes = {
            "window": 2,
            "n_columns": 3000,
            "directional": True,
            "stoplist": "full",
            "subsample": 1e-5,
            "cleaning": Cleaning(breaks="sentence"),
        }
        keys = {base.key()}
        for name, value in changes.items():
            keys.add(dataclasses.replace(base, **{name: value}).key())
        self.assertEqual(len(keys), len(changes) + 1)
        self.assertEqual(len(base.key()), 12)

    def test_stoplists(self):
        ranked = [f"w{i}" for i in range(200)]
        self.assertIn("the", load_stoplist("base30", ranked))
        self.assertIn("whose", load_stoplist("full", ranked))
        top = load_stoplist("top100", ranked)
        self.assertEqual(len(top), 100)
        self.assertIn("w99", top)
        self.assertNotIn("w100", top)


class ParquetJobsTest(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(folder.cleanup)
        self.folder = folder.name
        self.first = os.path.join(folder.name, "first.parquet")
        self.second = os.path.join(folder.name, "second.parquet")
        long_word = "x" * 40
        pq.write_table(
            pa.table(
                {
                    "text": [LINE, f"{LINE} {long_word} rare", None, LINE, LINE],
                    "url": ["u0", "u1", "u2", "u3", "u4"],
                    "token_count": [10, 10, 10, 10, 10],
                }
            ),
            self.first,
            row_group_size=2,
        )
        pq.write_table(
            pa.table(
                {
                    "text": ["blue blue red red green", LINE],
                    "url": ["u0", "v1"],
                    "token_count": [30, 10],
                }
            ),
            self.second,
            row_group_size=1,
        )
        self.jobs = [(self.first, g) for g in range(3)]
        self.jobs += [(self.second, g) for g in range(2)]
        self.words = ["red", "green", "blue"]
        self.settings = CountSettings(window=2, n_columns=3)
        self.checks = os.path.join(self.folder, "checks")

    def count(self, workers=0, **kwargs):
        options = {
            "milestones": [],
            "checkpoint_dir": self.checks,
            "on_milestone": None,
        }
        options.update(kwargs)
        return count_table(self.jobs, self.words, self.settings, workers, **options)

    def other_dir(self, name):
        return os.path.join(self.folder, name)

    def test_vocabulary_rejects_long_tokens_and_floor(self):
        result = vocabulary_pass(self.jobs, Cleaning(), 0, floor=2)
        self.assertNotIn("x" * 40, result.counts)
        self.assertNotIn("rare", result.counts)
        self.assertEqual(result.counts["red"], 12)
        self.assertEqual(result.jobs_used, self.jobs)

    def test_vocabulary_stops_at_job_boundary_past_max_words(self):
        result = vocabulary_pass(self.jobs, Cleaning(), 0, floor=1, max_words=6)
        self.assertEqual(result.jobs_used, self.jobs[:1])
        self.assertEqual(result.total_words, 14)

    def test_vocabulary_prune_keeps_common_counts(self):
        whole = vocabulary_pass(self.jobs, Cleaning(), 0, floor=2)
        pruned = vocabulary_pass(self.jobs, Cleaning(), 0, floor=2, max_entries=2)
        self.assertEqual(pruned.counts, whole.counts)

    def test_vocabulary_pool_equals_single_process(self):
        one = vocabulary_pass(self.jobs, Cleaning(), 0, floor=2)
        two = vocabulary_pass(self.jobs, Cleaning(), 2, floor=2)
        self.assertEqual(one, two)

    def test_pool_equals_single_process(self):
        one = self.count(workers=0, checkpoint_dir=self.other_dir("a"))
        two = self.count(workers=2, checkpoint_dir=self.other_dir("b"))
        self.assertTrue(np.array_equal(one, two))
        self.assertGreater(int(one.sum()), 0)

    def test_milestones_fire_and_table_grows(self):
        seen = []
        final = self.count(
            milestones=[6, 20],
            on_milestone=lambda name, words, table: seen.append(
                (name, words, int(table.sum()))
            ),
        )
        self.assertEqual([item[:2] for item in seen], [(6, 14), (20, 20)])
        self.assertLess(seen[0][2], seen[1][2])
        self.assertLessEqual(seen[1][2], int(final.sum()))

    def test_checkpoint_round_trip(self):
        final = self.count(milestones=[6])
        got = load_checkpoint(
            self.checks, self.settings, words_hash(self.words), "final"
        )
        self.assertTrue(np.array_equal(got, final))

    def test_stale_checkpoint_not_loaded(self):
        self.count()
        other = dataclasses.replace(self.settings, window=1)
        hash_of_words = words_hash(self.words)
        self.assertIsNone(load_checkpoint(self.checks, other, hash_of_words, "final"))
        self.assertIsNone(load_checkpoint(self.checks, self.settings, "x", "final"))

    def test_partial_checkpoint_not_loaded(self):
        hash_of_words = words_hash(self.words)
        base = checkpoint_name(self.checks, self.settings, hash_of_words, "final")
        os.makedirs(self.checks)
        with open(base + ".npy.tmp", "wb") as f:
            f.write(b"half a table")
        self.assertIsNone(
            load_checkpoint(self.checks, self.settings, hash_of_words, "final")
        )
        whole = self.count(checkpoint_dir=self.other_dir("whole"))
        self.assertTrue(np.array_equal(self.count(), whole))

    def test_table_with_no_state_not_loaded(self):
        self.count()
        hash_of_words = words_hash(self.words)
        base = checkpoint_name(self.checks, self.settings, hash_of_words, "final")
        os.remove(base + ".json")
        self.assertIsNone(
            load_checkpoint(self.checks, self.settings, hash_of_words, "final")
        )

    def test_failed_overwrite_leaves_no_usable_checkpoint(self):
        base = os.path.join(self.checks, "name")
        table = np.zeros((1, 1), dtype=np.uint64)
        save_checkpoint(base, table, {"jobs": "old", "done": 1, "words": 1})
        with mock.patch("counting.np.save", side_effect=OSError):
            with self.assertRaises(OSError):
                save_checkpoint(base, table, {"jobs": "new", "done": 2, "words": 2})
        self.assertIsNone(read_state(base))

    def test_resume_equals_uninterrupted(self):
        whole = self.count(checkpoint_dir=self.other_dir("whole"))

        class Crash(Exception):
            pass

        def crash(name, words, table):
            raise Crash

        with self.assertRaises(Crash):
            self.count(milestones=[6], on_milestone=crash)
        hash_of_words = words_hash(self.words)
        self.assertIsNotNone(
            load_checkpoint(self.checks, self.settings, hash_of_words, 6)
        )
        self.assertTrue(np.array_equal(self.count(milestones=[6]), whole))

    def test_resume_reports_the_loaded_checkpoint_again(self):
        class Crash(Exception):
            pass

        def crash(name, words, table):
            raise Crash

        with self.assertRaises(Crash):
            self.count(milestones=[6, 20], on_milestone=crash)
        seen = []
        self.count(
            milestones=[6, 20],
            on_milestone=lambda name, words, table: seen.append((name, words)),
        )
        self.assertEqual(seen, [(6, 14), (20, 20)])

    def test_changed_jobs_do_not_resume(self):
        self.count(milestones=[6])
        self.jobs = self.jobs[:1]
        fewer = self.count(milestones=[6])
        fresh = self.count(checkpoint_dir=self.other_dir("fresh"))
        self.assertTrue(np.array_equal(fewer, fresh))

    def test_rejects_table_over_limit(self):
        self.settings = CountSettings(n_columns=20000, directional=True)
        with self.assertRaises(ValueError):
            self.count()

    def test_wide_table_is_uint32_and_holds_the_same_counts(self):
        counts = {"red": 1000, "green": 1000, "blue": 1000}
        self.settings = CountSettings(window=2, n_columns=15000)
        wide = self.count(counts=counts)
        self.settings = CountSettings(window=2, n_columns=3)
        narrow = self.count(checkpoint_dir=self.other_dir("narrow"))
        self.assertEqual(wide.dtype, np.uint32)
        self.assertEqual(wide.shape, (3, 15000))
        self.assertEqual(narrow.dtype, np.uint64)
        self.assertTrue(np.array_equal(wide[:, :3], narrow))
        self.assertEqual(int(wide[:, 3:].sum()), 0)

    def test_wide_table_needs_counts(self):
        self.settings = CountSettings(window=2, n_columns=15000)
        with self.assertRaisesRegex(ValueError, "word counts"):
            self.count()

    def test_wide_table_bound_is_exact_at_32_bits(self):
        self.settings = CountSettings(window=2, n_columns=15000)
        fits = {"red": 2**32 // 6, "green": 5, "blue": 5}
        self.assertEqual(self.count(counts=fits).dtype, np.uint32)
        over = {"red": 2**32 // 6 + 1, "green": 5, "blue": 5}
        with self.assertRaisesRegex(ValueError, "32 bits"):
            self.count(counts=over, checkpoint_dir=self.other_dir("over"))

    def test_wide_table_bound_ignores_words_without_a_column(self):
        self.settings = CountSettings(window=2, n_columns=15000, stoplist="full")
        stop = sorted(load_stoplist("full", []))[0]
        self.words = ["red", "green", stop]
        counts = {"red": 5, "green": 5, stop: 2**31}
        self.assertEqual(self.count(counts=counts).dtype, np.uint32)

    def test_subsample_is_deterministic(self):
        self.settings = dataclasses.replace(self.settings, subsample=1e-2)
        counts = {"red": 1000, "green": 1000, "blue": 1000}
        one = self.count(counts=counts, checkpoint_dir=self.other_dir("s1"))
        two = self.count(counts=counts, checkpoint_dir=self.other_dir("s2"))
        self.assertTrue(np.array_equal(one, two))
        self.settings = dataclasses.replace(self.settings, subsample=None)
        full = self.count(checkpoint_dir=self.other_dir("s3"))
        self.assertLess(int(one.sum()), int(full.sum()))

    def test_subsample_needs_counts(self):
        self.settings = dataclasses.replace(self.settings, subsample=1e-2)
        with self.assertRaises(ValueError):
            self.count()

    def test_dead_worker_stops_the_run(self):
        out = os.path.join(self.folder, "out.txt")
        with open(out, "w", encoding="utf-8") as f:
            subprocess.run(
                [sys.executable, "-c", DEAD_WORKER],
                cwd=ROOT,
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=60,
            )
        self.assertIn("stopped", fileio.read_text(out))

    def test_overlap_measure(self):
        result = measure_overlap([self.first], [self.second])
        self.assertEqual(result["shared_rows"], 1)
        self.assertEqual(result["fine_rows"], 2)
        self.assertAlmostEqual(result["shared_token_share"], 0.75)

    def test_skip_urls_drop_only_listed_paths(self):
        overlap = measure_overlap([self.first], [self.second])
        kept = self.count(checkpoint_dir=self.other_dir("x"))
        dropped = self.count(
            checkpoint_dir=self.other_dir("y"), skip_urls=overlap["skip_urls"]
        )
        self.assertLess(int(dropped.sum()), int(kept.sum()))
        self.jobs = [(self.first, 0)]
        only_uk = self.count(checkpoint_dir=self.other_dir("z"))
        only_uk_skip = self.count(
            checkpoint_dir=self.other_dir("w"), skip_urls=overlap["skip_urls"]
        )
        self.assertTrue(np.array_equal(only_uk, only_uk_skip))


if __name__ == "__main__":
    unittest.main()
