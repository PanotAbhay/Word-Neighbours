import dataclasses
import glob
import json
import os
import tempfile
import unittest
import weakref
from concurrent.futures.process import BrokenProcessPool
from unittest import mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import fileio
import tune
from counting import MAX_TABLE_COLUMNS, MAX_UINT64_COLUMNS, CountSettings
from model import load_model, odd_one_out
from scoring import TaggedPuzzle, TestRefused, model_hash, run_test
from tune import (
    COUNT_GRID,
    FULL_WORDS,
    MAX_ROWS,
    METHOD_GRID,
    SMALL_DIMS,
    Paths,
    Runner,
    State,
    TrainError,
    VECTOR_GRID,
    append_row,
    better,
    bits_for,
    fold_scores,
    freeze_decision,
    keeps,
    measure_only,
    method_round,
    pick_finalist,
    run_key,
    run_method_round,
    ship_milestone,
    sweep_settings,
    text_files,
    train_corpus,
    train_full,
    tune_folds,
    vector_key,
    zipf3_coverage,
)
from tests.test_model import RULE_VECTORS, RULE_WORDS, VECTORS, WORDS
from textread import group_order, list_groups
from vectors import VectorSettings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRUIT = "apple pear plum cherry peach mango"
FURNITURE = "chair sofa table desk stool bench"
TAGS = (
    "part=noun type=unrelated target=fruit odd=furniture domain=food form=base pair=-"
)


class RulesTest(unittest.TestCase):
    def test_freeze_decision(self):
        self.assertEqual(freeze_decision(0.92), "freeze")
        self.assertEqual(freeze_decision(0.90), "freeze")
        self.assertEqual(freeze_decision(0.899), "stall")

    def test_ship_full_unless_clearly_lower(self):
        self.assertEqual(
            ship_milestone({100: 0.80, 1000: 0.85, 9000: 0.84}, 0.02), 9000
        )
        self.assertEqual(
            ship_milestone({100: 0.80, 1000: 0.85, 9000: 0.80}, 0.02), 1000
        )

    def test_ship_only_one_point(self):
        self.assertEqual(ship_milestone({5000: 0.5}, 0.02), 5000)

    def test_better_needs_gain(self):
        self.assertFalse(better(0.804, 0.80))
        self.assertTrue(better(0.806, 0.80))

    def test_grids_hold_planned_values(self):
        grid = dict(COUNT_GRID)
        self.assertEqual(list(grid), [
            "stoplist", "window", "n_columns", "directional",
            "breaks", "apostrophes", "accents", "subsample",
        ])  # fmt: skip
        self.assertEqual(grid["n_columns"], [3000, 5000, 10000])
        self.assertEqual(dict(VECTOR_GRID)["p"], [0.0, 0.5, 1.0])

    def test_method_grid_holds_planned_values(self):
        self.assertEqual(
            METHOD_GRID,
            [
                ("p", [-0.5, -0.25]),
                ("skip", [1, 2, 5, 10]),
                ("shift", [2.0, 5.0]),
                ("smoothing", [0.5, 1.0]),
                ("dims", [400, 500, 600]),
                ("rule", ["median", "max"]),
                ("p", [-0.5, -0.25, 0.0]),
            ],
        )
        self.assertEqual(SMALL_DIMS, 300)

    def test_tune_module_never_names_the_test_file(self):
        text = fileio.read_text(os.path.join(ROOT, "tune.py"))
        self.assertNotIn("puzzles_test", text)
        self.assertNotIn("test_log", text)


class FakeMeasure:
    def __init__(self, scores=None, seconds_per_word=1e-5, clock=None):
        self.scores = scores or {}
        self.seconds_per_word = seconds_per_word
        self.clock = clock
        self.calls = []

    def __call__(self, settings, vector, words):
        self.calls.append((settings, vector, words))
        result = {
            "tune": 0.70,
            "folds": [0.70, 0.70],
            "dev": 0.5,
            "probes": {"easy": 0.5},
            "seconds": 0.0,
        }
        for check, values in self.scores.items():
            if check(settings, vector, words):
                result.update(values)
        result["seconds"] = self.seconds_per_word * words
        if self.clock is not None:
            self.clock.now += result["seconds"]
        return result


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class SweepTest(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(folder.cleanup)
        self.log = os.path.join(folder.name, "tune_log.jsonl")

    def test_probes_never_choose(self):
        scores = {
            (lambda s, v, w: s.window == 2): {
                "tune": 0.60,
                "dev": 0.99,
                "probes": {"easy": 0.99},
            }
        }
        settings, _ = sweep_settings(FakeMeasure(scores), 1e12)
        self.assertEqual(settings.window, 5)

    def test_tune_score_chooses(self):
        scores = {(lambda s, v, w: s.window == 2): {"tune": 0.80, "dev": 0.01}}
        settings, _ = sweep_settings(FakeMeasure(scores), 1e12)
        self.assertEqual(settings.window, 2)

    def test_small_gain_does_not_change_setting(self):
        scores = {(lambda s, v, w: s.window == 2): {"tune": 0.703}}
        settings, _ = sweep_settings(FakeMeasure(scores), 1e12)
        self.assertEqual(settings.window, 5)

    def test_vector_grid_runs_at_large_prefix(self):
        scores = {(lambda s, v, w: v.p == 1.0): {"tune": 0.80}}
        measure = FakeMeasure(scores)
        _, vector = sweep_settings(measure, 1e12)
        self.assertEqual(vector.p, 1.0)
        self.assertEqual({w for _, v, w in measure.calls if v.p != 0.5}, {1000000000})

    def test_no_setting_over_table_limit(self):
        scores = {(lambda s, v, w: s.directional): {"tune": 0.80}}
        measure = FakeMeasure(scores)
        settings, _ = sweep_settings(measure, 1e12)
        self.assertTrue(settings.directional)
        self.assertEqual(settings.n_columns, 5000)
        self.assertGreater(len(measure.calls), 5)
        for tried, _, _ in measure.calls:
            self.assertLessEqual(tried.table_columns(), MAX_UINT64_COLUMNS)

    def test_budget_stops_sweep_and_reserves_two_full_counts(self):
        clock = FakeClock()
        measure = FakeMeasure(clock=clock)
        full = 1e-5 * FULL_WORDS
        budget = 2 * full + 30000
        sweep_settings(measure, budget, clock=clock)
        self.assertLessEqual(clock.now, budget - 2 * full)
        unlimited = FakeMeasure()
        sweep_settings(unlimited, 1e12)
        self.assertLess(len(measure.calls), len(unlimited.calls))
        self.assertGreater(len(measure.calls), 0)

    def test_resume_uses_tune_log(self):
        first = FakeMeasure({(lambda s, v, w: s.window == 2): {"tune": 0.80}})
        one = sweep_settings(first, 1e12, log_path=self.log)
        again = FakeMeasure()
        two = sweep_settings(again, 1e12, log_path=self.log)
        self.assertEqual(again.calls, [])
        self.assertEqual(one, two)

    def test_log_lines_hold_settings_and_scores(self):
        sweep_settings(FakeMeasure(), 1e12, log_path=self.log)
        rows = [json.loads(line) for line in fileio.read_lines(self.log)]
        self.assertIn("settings", rows[0])
        self.assertIn("vector", rows[0])
        self.assertEqual(rows[0]["words"], 100000000)
        self.assertEqual(rows[0]["tune"], 0.70)

    def test_no_budget_returns_start_settings(self):
        measure = FakeMeasure()
        settings, vector = sweep_settings(measure, 0)
        self.assertEqual(settings, CountSettings())
        self.assertEqual(vector, VectorSettings())
        self.assertEqual(measure.calls, [])


class VectorKeyTest(unittest.TestCase):
    def test_new_fields_at_their_defaults_leave_the_old_key_unchanged(self):
        self.assertEqual(
            vector_key(VectorSettings(p=0.0)),
            '{"centre": true, "dims": 300, "p": 0.0, "smoothing": 0.75}',
        )

    def test_changed_new_fields_enter_the_key(self):
        base = vector_key(VectorSettings())
        for change in ({"skip": 2}, {"shift": 2.0}, {"rule": "median"}):
            key = vector_key(VectorSettings(**change))
            ((name, value),) = change.items()
            self.assertNotEqual(key, base)
            self.assertIn(f'"{name}": {json.dumps(value)}', key)

    def test_different_shifts_give_different_keys(self):
        self.assertNotEqual(
            vector_key(VectorSettings(shift=2.0)),
            vector_key(VectorSettings(shift=5.0)),
        )


class FoldTest(unittest.TestCase):
    TARGETS = [f"class{i}" for i in range(10) for _ in range(3)]

    def test_same_class_gets_the_same_fold(self):
        targets = ["a", "b", "a", "c", "b", "a"]
        folds = tune_folds(targets)
        self.assertEqual(folds[0], folds[2])
        self.assertEqual(folds[0], folds[5])
        self.assertEqual(folds[1], folds[4])
        self.assertEqual(folds, tune_folds(targets))

    def test_classes_split_into_two_equal_halves(self):
        folds = tune_folds(self.TARGETS)
        self.assertEqual([folds.count(0), folds.count(1)], [15, 15])

    def test_split_ignores_puzzle_order(self):
        forward = dict(zip(self.TARGETS, tune_folds(self.TARGETS)))
        backward = dict(zip(self.TARGETS[::-1], tune_folds(self.TARGETS[::-1])))
        self.assertEqual(forward, backward)

    def test_fold_scores_average_each_fold(self):
        self.assertEqual(
            fold_scores([True, False, True, True], [0, 0, 1, 1]), [0.5, 1.0]
        )

    def test_empty_fold_scores_zero(self):
        self.assertEqual(fold_scores([True], [0]), [1.0, 0.0])


def outcome(tune, folds):
    return {"tune": tune, "folds": folds}


class KeepsTest(unittest.TestCase):
    OLD = outcome(0.80, [0.80, 0.80])

    def test_gain_on_both_folds_is_kept(self):
        self.assertTrue(keeps(outcome(0.81, [0.82, 0.80]), self.OLD))

    def test_small_gain_is_not_kept(self):
        self.assertFalse(keeps(outcome(0.803, [0.81, 0.81]), self.OLD))

    def test_gain_that_hurts_one_fold_is_not_kept(self):
        self.assertFalse(keeps(outcome(0.85, [0.95, 0.75]), self.OLD))

    def test_equal_fold_is_fine(self):
        self.assertTrue(keeps(outcome(0.85, [0.90, 0.80]), self.OLD))


class MethodRoundTest(unittest.TestCase):
    START = (CountSettings(), VectorSettings(p=0.0))
    SIZE = 1000
    GRID = [("p", [-0.5, -0.25]), ("skip", [1, 2]), ("dims", [400])]

    def setUp(self):
        folder = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(folder.cleanup)
        self.log = os.path.join(folder.name, "tune_log.jsonl")

    def run_round(self, measure, grid=None, **kwargs):
        grid = self.GRID if grid is None else grid
        return method_round(measure, self.START, self.SIZE, grid, **kwargs)

    def test_baseline_is_measured_first_then_each_value_in_order(self):
        measure = FakeMeasure()
        self.run_round(measure)
        seen = [(v.p, v.skip, v.dims) for _, v, _ in measure.calls]
        self.assertEqual(
            seen,
            [
                (0.0, 0, 300),
                (-0.5, 0, 300),
                (-0.25, 0, 300),
                (0.0, 1, 300),
                (0.0, 2, 300),
                (0.0, 0, 400),
            ],
        )
        self.assertEqual({w for _, _, w in measure.calls}, {self.SIZE})

    def test_kept_change_carries_into_the_next_axis(self):
        scores = {
            (lambda s, v, w: v.p == -0.5): {"tune": 0.75, "folds": [0.75, 0.75]},
        }
        measure = FakeMeasure(scores)
        got = self.run_round(measure)
        self.assertEqual(got.best.vector.p, -0.5)
        later = [v for _, v, _ in measure.calls if v.skip or v.dims != 300]
        self.assertTrue(later)
        self.assertEqual({v.p for v in later}, {-0.5})

    def test_gain_on_one_fold_only_is_rejected(self):
        scores = {
            (lambda s, v, w: v.p == -0.5): {"tune": 0.80, "folds": [0.90, 0.60]},
        }
        got = self.run_round(FakeMeasure(scores))
        self.assertEqual(got.best.vector.p, 0.0)
        self.assertFalse(got.history[0]["kept"])

    def test_small_gain_is_rejected(self):
        scores = {
            (lambda s, v, w: v.p == -0.5): {"tune": 0.703, "folds": [0.71, 0.71]},
        }
        got = self.run_round(FakeMeasure(scores))
        self.assertEqual(got.best.vector.p, 0.0)

    def test_later_value_compares_with_the_updated_best(self):
        scores = {
            (lambda s, v, w: v.p == -0.5): {"tune": 0.75, "folds": [0.75, 0.75]},
            (lambda s, v, w: v.p == -0.25): {"tune": 0.754, "folds": [0.76, 0.75]},
        }
        got = self.run_round(FakeMeasure(scores))
        self.assertEqual(got.best.vector.p, -0.5)

    def test_current_value_is_skipped(self):
        measure = FakeMeasure()
        self.run_round(measure, [("p", [0.0, -0.5])])
        self.assertEqual([v.p for _, v, _ in measure.calls], [0.0, -0.5])

    def test_count_axis_keeps_to_the_column_limit(self):
        scores = {
            (lambda s, v, w: s.n_columns == 20000): {
                "tune": 0.80,
                "folds": [0.80, 0.80],
            },
        }
        measure = FakeMeasure(scores)
        got = self.run_round(measure, [("n_columns", [20000, 25000])])
        self.assertEqual([s.n_columns for s, _, _ in measure.calls], [10000, 20000])
        self.assertEqual(got.best.settings.n_columns, 20000)
        for tried, _, _ in measure.calls:
            self.assertLessEqual(tried.table_columns(), MAX_TABLE_COLUMNS)

    def test_history_lists_every_comparison(self):
        got = self.run_round(FakeMeasure())
        self.assertEqual(
            [(h["name"], h["value"]) for h in got.history],
            [("p", -0.5), ("p", -0.25), ("skip", 1), ("skip", 2), ("dims", 400)],
        )
        self.assertEqual(
            got.history[0],
            {
                "name": "p",
                "value": -0.5,
                "tune": 0.70,
                "folds": [0.70, 0.70],
                "kept": False,
            },
        )

    def test_dev_and_probes_never_choose(self):
        scores = {
            (lambda s, v, w: v.p == -0.5): {
                "dev": 0.99,
                "probes": {"easy": 0.99},
            },
        }
        got = self.run_round(FakeMeasure(scores))
        self.assertEqual(got.best.vector.p, 0.0)

    def test_small_is_the_last_accepted_state_with_few_dims(self):
        scores = {
            (lambda s, v, w: v.p == -0.5): {"tune": 0.75, "folds": [0.75, 0.75]},
            (lambda s, v, w: v.p == -0.5 and v.dims == 400): {
                "tune": 0.80,
                "folds": [0.80, 0.80],
            },
        }
        got = self.run_round(FakeMeasure(scores), [("p", [-0.5]), ("dims", [400])])
        self.assertEqual(got.best.vector, VectorSettings(p=-0.5, dims=400))
        self.assertEqual(got.small.vector, VectorSettings(p=-0.5))
        self.assertEqual(got.small.result["tune"], 0.75)

    def test_small_is_the_baseline_when_nothing_small_is_kept(self):
        scores = {
            (lambda s, v, w: v.dims == 400): {"tune": 0.80, "folds": [0.80, 0.80]},
        }
        got = self.run_round(FakeMeasure(scores), [("dims", [400])])
        self.assertEqual(got.best.vector.dims, 400)
        self.assertEqual(got.small.vector, self.START[1])

    def test_log_rows_hold_the_folds_and_the_vector(self):
        self.run_round(FakeMeasure(), log_path=self.log)
        rows = [json.loads(line) for line in fileio.read_lines(self.log)]
        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[0]["folds"], [0.70, 0.70])
        self.assertEqual(rows[0]["words"], self.SIZE)
        self.assertEqual(rows[0]["vector"]["rule"], "sum")

    def test_resume_uses_the_log(self):
        scores = {
            (lambda s, v, w: v.p == -0.5): {"tune": 0.75, "folds": [0.75, 0.75]},
        }
        one = self.run_round(FakeMeasure(scores), log_path=self.log)
        again = FakeMeasure()
        two = self.run_round(again, log_path=self.log)
        self.assertEqual(again.calls, [])
        self.assertEqual(
            (one.best.settings, one.best.vector, one.history),
            (two.best.settings, two.best.vector, two.history),
        )
        self.assertEqual(one.small.vector, two.small.vector)

    def test_cached_row_without_folds_is_measured_again(self):
        settings, vector = self.START
        old = {"tune": 0.70, "dev": 0.5, "probes": {}, "seconds": 0.0}
        append_row(self.log, {"key": run_key(settings, vector, self.SIZE), **old})
        measure = FakeMeasure()
        self.run_round(measure, log_path=self.log)
        self.assertEqual(measure.calls[0][:2], self.START)
        self.assertEqual(len(measure.calls), 6)

    def test_no_budget_returns_the_start_unmeasured(self):
        measure = FakeMeasure()
        got = self.run_round(measure, budget_seconds=0)
        self.assertEqual(measure.calls, [])
        self.assertEqual((got.best.settings, got.best.vector), self.START)
        self.assertIsNone(got.best.result)
        self.assertIsNone(got.small)
        self.assertEqual(got.history, [])

    def test_budget_stops_the_round(self):
        clock = FakeClock()
        measure = FakeMeasure(seconds_per_word=1e-3, clock=clock)
        self.run_round(measure, budget_seconds=2.5, clock=clock)
        self.assertEqual(len(measure.calls), 2)
        self.assertLessEqual(clock.now, 2.5)


class ShipTest(unittest.TestCase):
    @staticmethod
    def state(dims, tune):
        return State(CountSettings(), VectorSettings(dims=dims), {"tune": tune})

    def test_wide_model_ships_when_the_stored_score_holds(self):
        best, small = self.state(600, 0.88), self.state(300, 0.86)
        self.assertIs(pick_finalist(best, small, 0.87), best)
        self.assertIs(pick_finalist(best, small, 0.86), best)

    def test_small_model_ships_when_half_precision_loses_the_gain(self):
        best, small = self.state(600, 0.88), self.state(300, 0.86)
        self.assertIs(pick_finalist(best, small, 0.85), small)

    def test_small_best_ships_as_it_is(self):
        best = self.state(300, 0.88)
        self.assertIs(pick_finalist(best, self.state(300, 0.86), 0.10), best)

    def test_no_small_state_ships_the_best(self):
        best = self.state(600, 0.88)
        self.assertIs(pick_finalist(best, None, 0.10), best)

    def test_bits_follow_the_dimension_limit(self):
        self.assertEqual(bits_for(VectorSettings(dims=300)), 32)
        self.assertEqual(bits_for(VectorSettings(dims=301)), 16)
        self.assertEqual(bits_for(VectorSettings(dims=600)), 16)


class CorpusTest(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(folder.cleanup)
        self.folder = folder.name

    def train_tiny(self, vector):
        first = os.path.join(self.folder, "a.txt")
        second = os.path.join(self.folder, "b.txt")
        fruit = FRUIT.replace(" ", " and ")
        furniture = FURNITURE.replace(" ", " and ")
        fileio.write_lines(first, [f"{fruit}. {fruit}. {fruit}."] * 10)
        fileio.write_lines(second, [f"{furniture}. {furniture}. {furniture}."] * 10)
        out = os.path.join(self.folder, "model")
        train_corpus(
            [first, second], CountSettings(window=2, n_columns=20), vector, out
        )
        return load_model(out)

    def test_smoke_train_on_tiny_corpus(self):
        model = self.train_tiny(VectorSettings(dims=4))
        self.assertIn("apple", model.index)
        answer = odd_one_out(["apple", "pear", "plum", "chair"], model)
        self.assertEqual(answer, "chair")
        self.assertFalse(model.meta["frozen"])

    def test_corpus_model_keeps_the_rule_and_the_shift(self):
        with mock.patch("tune.ppmi_matrix", wraps=tune.ppmi_matrix) as ppmi:
            model = self.train_tiny(VectorSettings(dims=4, shift=2.0, rule="max"))
        self.assertEqual({call.args[2:] for call in ppmi.call_args_list}, {(2.0,)})
        self.assertEqual(model.meta["rule"], "max")
        self.assertEqual(model.meta["vector"]["shift"], 2.0)


def write_small_text(folder):
    uk = os.path.join(folder, "text", "uk")
    fine = os.path.join(folder, "text", "fine")
    os.makedirs(uk)
    os.makedirs(fine)
    lines = [FRUIT.replace(" ", " and "), FURNITURE.replace(" ", " and ")]
    for name, path, seed in (("u", uk, 0), ("f", fine, 1)):
        texts = [lines[(i + seed) % 2] for i in range(24)]
        table = pa.table(
            {
                "text": texts,
                "url": [f"{name}{i}" for i in range(24)],
                "token_count": [10] * 24,
            }
        )
        pq.write_table(table, os.path.join(path, f"{name}.parquet"), row_group_size=4)


class TrainFixture(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(folder.cleanup)
        self.folder = folder.name
        write_small_text(self.folder)
        self.tune = os.path.join(self.folder, "puzzles_tune.txt")
        fileio.write_lines(
            self.tune,
            [
                f"apple pear plum chair ; chair ; {TAGS}",
                f"cherry peach mango sofa ; sofa ; {TAGS}",
            ],
        )
        self.work = os.path.join(self.folder, "work")
        self.model = os.path.join(self.folder, "model")

    def paths(self, **kwargs):
        values = {
            "text": os.path.join(self.folder, "text"),
            "tune": self.tune,
            "dev": os.path.join(ROOT, "data", "puzzles_diagnostic.txt"),
            "probes": os.path.join(ROOT, "data", "probes.txt"),
        }
        values.update(kwargs)
        return Paths(**values)

    def train(self, **kwargs):
        options = {
            "paths": self.paths(),
            "workers": 0,
            "sweep_words": (20, 40),
            "milestones": (30, 60),
            "start_settings": CountSettings(window=2, n_columns=12),
            "start_vector": VectorSettings(dims=3),
        }
        options.update(kwargs)
        with mock.patch("tune.COUNT_GRID", [("window", [2, 5])]):
            with mock.patch("tune.VECTOR_GRID", [("p", [0.5, 1.0])]):
                return train_full(self.work, self.model, **options)

    def close_puzzle_answered_apple(self):
        tags = TAGS.replace("type=unrelated", "type=close")
        fileio.write_lines(self.tune, [f"apple pear plum chair ; apple ; {tags}"])


class FullTrainTest(TrainFixture):
    def test_full_train_saves_model_with_meta(self):
        result = self.train()
        model = load_model(self.model)
        self.assertIn("apple", model.index)
        self.assertTrue(model.meta["frozen"])
        self.assertEqual(model.meta["stalled"], result["decision"] == "stall")
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["decision"], "freeze")
        for key in (
            "settings",
            "vector",
            "rule",
            "curve",
            "tune_report",
            "coverage",
            "overlap",
        ):
            self.assertIn(key, model.meta)
        self.assertGreaterEqual(len(model.meta["curve"]), 2)

    def test_full_train_records_time_and_log(self):
        self.train()
        with open(os.path.join(self.work, "time_used.json"), encoding="utf-8") as f:
            used = json.load(f)
        self.assertGreater(used["train"], 0.0)
        rows = fileio.read_lines(os.path.join(self.work, "tune_log.jsonl"))
        self.assertGreater(len(rows), 3)

    def test_full_train_low_score_with_no_budget_is_stall(self):
        fileio.write_lines(self.tune, [f"apple pear plum chair ; apple ; {TAGS}"])
        result = self.train(budget_hours=0)
        self.assertEqual(result["decision"], "stall")
        self.assertTrue(load_model(self.model).meta["stalled"])

    def test_full_train_low_score_with_budget_left_is_stall(self):
        fileio.write_lines(self.tune, [f"apple pear plum chair ; apple ; {TAGS}"])
        result = self.train(budget_hours=1000)
        self.assertEqual(result["decision"], "stall")
        self.assertTrue(load_model(self.model).meta["stalled"])

    def test_stalled_model_takes_the_test_only_with_approval(self):
        fileio.write_lines(self.tune, [f"apple pear plum chair ; apple ; {TAGS}"])
        self.train()
        puzzles = os.path.join(self.folder, "held_back.txt")
        fileio.write_lines(puzzles, [f"apple pear plum sofa ; sofa ; {TAGS}"])
        log = os.path.join(self.folder, "log.txt")
        with self.assertRaisesRegex(TestRefused, "owner"):
            run_test(self.model, puzzles, log)
        self.assertFalse(os.path.exists(log))
        report = run_test(self.model, puzzles, log, owner_approved=True)
        self.assertIn("overall", report)
        self.assertTrue(os.path.exists(log))

    def test_hubness_kept_only_when_it_gains(self):
        self.close_puzzle_answered_apple()
        self.train()
        apple = load_model(self.model).index["apple"]

        def pulls_apple_out(vectors):
            hub = np.zeros(len(vectors))
            hub[apple] = 100.0
            return hub

        with mock.patch("tune.hub_scores", pulls_apple_out):
            result = self.train()
        model = load_model(self.model)
        self.assertEqual(result["decision"], "freeze")
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(model.hub[apple], 100.0)
        self.assertEqual(model.meta["hubness"]["tune_before"], 0.0)
        self.assertEqual(model.meta["hubness"]["tune_after"], 1.0)
        self.assertTrue(model.meta["hubness"]["kept"])

    def test_hubness_dropped_when_it_does_not_gain(self):
        self.close_puzzle_answered_apple()
        with mock.patch("tune.hub_scores", lambda vectors: np.zeros(len(vectors))):
            result = self.train()
        model = load_model(self.model)
        self.assertEqual(result["decision"], "stall")
        self.assertIsNone(model.hub)
        self.assertFalse(model.meta["hubness"]["kept"])

    def test_hubness_not_tried_when_close_score_is_high(self):
        tags = TAGS.replace("type=unrelated", "type=close")
        fileio.write_lines(self.tune, [f"apple pear plum chair ; chair ; {tags}"])
        with mock.patch("tune.hub_scores") as hub:
            self.train()
        hub.assert_not_called()
        self.assertIsNone(load_model(self.model).meta["hubness"])

    def test_crash_while_scoring_a_milestone_keeps_its_name(self):
        with mock.patch("tune.Runner.score", side_effect=RuntimeError):
            with self.assertRaises(RuntimeError):
                self.train(budget_hours=0)
        self.train(budget_hours=0)
        log = os.path.join(self.work, "tune_log.jsonl")
        rows = [json.loads(line) for line in fileio.read_lines(log)]
        found = {str(r["milestone"]): r["words"] for r in rows if "milestone" in r}
        self.assertEqual(sorted(found), ["30", "60", "final"])
        for name in ("30", "60"):
            state = glob.glob(os.path.join(self.work, "checks", f"*_{name}.json"))
            with open(state[0], encoding="utf-8") as f:
                self.assertEqual(found[name], json.load(f)["words"])

    def test_large_objects_are_released_before_the_next_big_step(self):
        tables, matrices, at_count, at_embed, at_load = [], [], [], [], []
        real_count, real_ppmi = tune.count_table, tune.ppmi_matrix
        real_embed, real_load = tune.embed, tune.load_checkpoint

        def dead(refs):
            return all(ref() is None for ref in refs)

        def count(*args, **kwargs):
            if args[5].endswith("checks"):
                at_count.append(dead(matrices))
            table = real_count(*args, **kwargs)
            tables.append(weakref.ref(table))
            return table

        def ppmi(table, smoothing, shift):
            matrix = real_ppmi(table, smoothing, shift)
            matrices.append(weakref.ref(matrix))
            return matrix

        def embed(matrix, vector):
            at_embed.append(dead(tables))
            return real_embed(matrix, vector)

        def load(*args):
            at_load.append(dead(tables) and dead(matrices))
            return real_load(*args)

        with (
            mock.patch("tune.count_table", count),
            mock.patch("tune.ppmi_matrix", ppmi),
            mock.patch("tune.embed", embed),
            mock.patch("tune.load_checkpoint", load),
            mock.patch("tune.ship_milestone", lambda curve, width: min(curve)),
        ):
            self.train()
        self.assertEqual(at_count, [True])
        self.assertTrue(all(at_embed))
        self.assertEqual(at_load, [True])

    def test_new_tune_file_measures_again(self):
        log = os.path.join(self.work, "tune_log.jsonl")

        def sweep_rows():
            rows = [json.loads(line) for line in fileio.read_lines(log)]
            return sum("key" in row for row in rows)

        self.train()
        first = sweep_rows()
        self.train()
        self.assertEqual(sweep_rows(), first)
        fileio.write_lines(self.tune, [f"apple pear plum sofa ; sofa ; {TAGS}"])
        self.train()
        self.assertGreater(sweep_rows(), first)

    def text_paths(self, entries):
        sources = os.path.join(self.folder, "sources.json")
        with open(sources, "w", encoding="utf-8") as f:
            json.dump({"lists": [], "text": entries}, f)
        return self.paths(sources=sources)

    def text_entries(self):
        entries = []
        for group, name in (("uk", "u.parquet"), ("fine", "f.parquet")):
            path = os.path.join(self.folder, "text", group, name)
            size = os.path.getsize(path)
            entries.append({"group": group, "name": name, "size": size})
        return entries

    def test_complete_text_is_accepted(self):
        uk, fine = text_files(self.text_paths(self.text_entries()))
        self.assertEqual(len(uk) + len(fine), 2)

    def test_missing_text_file_is_train_error(self):
        entries = self.text_entries()
        entries.append({"group": "fine", "name": "g.parquet", "size": 1})
        with self.assertRaisesRegex(TrainError, "download"):
            text_files(self.text_paths(entries))

    def test_wrong_size_text_file_is_train_error(self):
        entries = self.text_entries()
        entries[0]["size"] += 1
        with self.assertRaisesRegex(TrainError, "download"):
            text_files(self.text_paths(entries))

    def test_missing_text_is_train_error(self):
        paths = self.paths(text=os.path.join(self.folder, "nothing"))
        with self.assertRaises(TrainError):
            self.train(paths=paths)

    def test_missing_tune_file_is_train_error(self):
        paths = self.paths(tune=os.path.join(self.folder, "nothing.txt"))
        with self.assertRaises(TrainError):
            self.train(paths=paths)

    def test_overlap_recorded_once(self):
        self.train()
        with mock.patch("tune.measure_overlap") as overlap:
            self.train()
        overlap.assert_not_called()

    def test_resume_keeps_milestone_names(self):
        self.train()
        checks = os.path.join(self.work, "checks")
        for path in glob.glob(os.path.join(checks, "*_60.*")):
            os.remove(path)
        for path in glob.glob(os.path.join(checks, "*_final.*")):
            os.remove(path)
        log = os.path.join(self.work, "tune_log.jsonl")
        rows = [json.loads(line) for line in fileio.read_lines(log)]
        kept = [r for r in rows if r.get("milestone") not in (60, "final")]
        fileio.write_lines(log, [json.dumps(r) for r in kept])
        self.train()
        rows = [json.loads(line) for line in fileio.read_lines(log)]
        names = sorted(str(r["milestone"]) for r in rows if "milestone" in r)
        self.assertEqual(names, ["30", "60", "final"])

    def test_vocabulary_keeps_only_row_words(self):
        with mock.patch("tune.MAX_ROWS", 2):
            self.train(budget_hours=0)
        found = glob.glob(os.path.join(self.work, "vocab_*.json"))
        self.assertEqual(len(found), 1)
        with open(found[0], encoding="utf-8") as f:
            self.assertEqual(len(json.load(f)["counts"]), 2)

    def test_measure_only_prints_rates(self):
        with mock.patch("builtins.print") as shown:
            measure_only(
                40,
                self.work,
                paths=self.paths(),
                workers=0,
                settings=CountSettings(window=2, n_columns=12),
                vector=VectorSettings(dims=3),
            )
        text = " ".join(str(call) for call in shown.call_args_list)
        for piece in ("words/s", "updates/s", "peak RAM", "tune"):
            self.assertIn(piece, text)

    def test_measure_only_uses_the_shift(self):
        with mock.patch("tune.ppmi_matrix", wraps=tune.ppmi_matrix) as ppmi:
            with mock.patch("builtins.print"):
                measure_only(
                    40,
                    self.work,
                    paths=self.paths(),
                    workers=0,
                    settings=CountSettings(window=2, n_columns=12),
                    vector=VectorSettings(dims=3, shift=2.0),
                )
        self.assertGreater(ppmi.call_count, 0)
        self.assertEqual({call.args[2:] for call in ppmi.call_args_list}, {(2.0,)})

    def test_shift_reaches_every_ppmi_call(self):
        with mock.patch("tune.ppmi_matrix", wraps=tune.ppmi_matrix) as ppmi:
            self.train(start_vector=VectorSettings(dims=3, shift=2.0))
        self.assertGreater(ppmi.call_count, 3)
        self.assertEqual({call.args[2:] for call in ppmi.call_args_list}, {(2.0,)})

    def test_rule_reaches_every_scoring_call_and_the_saved_model(self):
        rules = []
        real_rate = Runner.rate

        def rate(self, rows, vectors, hub=None, rule="sum"):
            rules.append(rule)
            return real_rate(self, rows, vectors, hub, rule)

        with mock.patch("tune.Runner.rate", rate):
            self.train(start_vector=VectorSettings(dims=3, rule="median"))
        model = load_model(self.model)
        self.assertGreater(len(rules), 3)
        self.assertEqual(set(rules), {"median"})
        self.assertEqual(model.meta["rule"], "median")
        self.assertEqual(model.meta["vector"]["rule"], "median")


class RunnerTest(TrainFixture):
    def real_runner(self):
        uk, fine = text_files(self.paths())
        jobs = group_order(list_groups(uk), list_groups(fine))
        return Runner(jobs, self.paths(), self.work, 0)

    def test_rate_reports_the_score_of_each_fold(self):
        other = TAGS.replace("target=fruit", "target=seat")
        fileio.write_lines(
            self.tune,
            [
                f"apple pear plum chair ; chair ; {TAGS}",
                f"apple pear plum chair ; apple ; {other}",
            ],
        )
        result, _ = Runner([], self.paths(), self.work, 0).rate(WORDS, VECTORS)
        self.assertEqual(result["tune"], 0.5)
        self.assertEqual(sorted(result["folds"]), [0.0, 1.0])

    def test_rate_applies_the_rule(self):
        fileio.write_lines(self.tune, [f"a b c d ; c ; {TAGS}"])
        runner = Runner([], self.paths(), self.work, 0)
        by_sum, _ = runner.rate(RULE_WORDS, RULE_VECTORS)
        by_median, _ = runner.rate(RULE_WORDS, RULE_VECTORS, rule="median")
        self.assertEqual(by_sum["tune"], 0.0)
        self.assertEqual(by_median["tune"], 1.0)

    def test_matrix_is_built_again_only_for_a_new_shift(self):
        runner = self.real_runner()
        settings = CountSettings(window=2, n_columns=12)
        with mock.patch("tune.ppmi_matrix", wraps=tune.ppmi_matrix) as ppmi:
            runner.matrix(settings, 0.75, 1.0, 40)
            runner.matrix(settings, 0.75, 1.0, 40)
            runner.matrix(settings, 0.75, 2.0, 40)
        self.assertEqual([call.args[2] for call in ppmi.call_args_list], [1.0, 2.0])

    def test_matrix_removes_its_own_sweep_folder(self):
        runner = self.real_runner()
        runner.matrix(CountSettings(window=2, n_columns=12), 0.75, 1.0, 40)
        self.assertFalse(os.path.exists(runner.sweep_dir))

    def test_matrix_keeps_the_checkpoints_of_the_counts_folder(self):
        runner = self.real_runner()
        runner.counts_dir = os.path.join(self.work, "checks")
        runner.matrix(CountSettings(window=2, n_columns=12), 0.75, 1.0, None)
        found = glob.glob(os.path.join(self.work, "checks", "*_final.json"))
        self.assertEqual(len(found), 1)

    def test_call_skips_the_diagnostic_when_no_size_is_marked(self):
        runner = self.real_runner()
        with mock.patch.object(Runner, "no_svd") as diagnostic:
            result = runner(
                CountSettings(window=2, n_columns=12), VectorSettings(dims=3), None
            )
        diagnostic.assert_not_called()
        self.assertNotIn("no_svd", result)


class MethodDriverTest(TrainFixture):
    def setUp(self):
        super().setUp()
        self.candidate = os.path.join(self.folder, "candidate")

    def drive(self, **kwargs):
        options = {
            "paths": self.paths(),
            "workers": 0,
            "grid": [("p", [1.0]), ("skip", [1])],
        }
        options.update(kwargs)
        return run_method_round(self.work, self.model, self.candidate, **options)

    def stall_then_fix_the_tune_puzzle(self):
        self.close_puzzle_answered_apple()
        self.train()
        fileio.write_lines(self.tune, [f"apple pear plum chair ; chair ; {TAGS}"])

    def test_round_without_gain_leaves_the_model_alone(self):
        self.train()
        before = model_hash(self.model)
        result = self.drive()
        self.assertFalse(result["replaced"])
        self.assertEqual(result["decision"], "freeze")
        self.assertEqual(model_hash(self.model), before)

    def test_round_reuses_the_saved_counts_and_writes_a_candidate(self):
        self.train()
        checks = os.path.join(self.work, "checks")
        before = sorted(os.listdir(checks))
        self.drive()
        self.assertEqual(sorted(os.listdir(checks)), before)
        self.assertTrue(os.path.exists(os.path.join(self.candidate, "vectors.npy")))

    def test_round_replaces_the_model_when_the_stored_score_gains(self):
        self.stall_then_fix_the_tune_puzzle()
        result = self.drive()
        model = load_model(self.model)
        self.assertTrue(result["replaced"])
        self.assertEqual(result["before"], 0.0)
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["decision"], "freeze")
        self.assertTrue(model.meta["frozen"])
        self.assertFalse(model.meta["stalled"])
        self.assertEqual(model.meta["rule"], "sum")
        self.assertEqual(model.meta["bits"], 32)
        self.assertEqual(model.meta["method_round"]["baseline"], 1.0)
        self.assertEqual(len(model.meta["method_round"]["history"]), 2)

    def test_wide_vectors_are_stored_in_half_precision(self):
        self.stall_then_fix_the_tune_puzzle()
        with mock.patch("tune.SMALL_DIMS", 2):
            self.drive()
        for folder in (self.candidate, self.model):
            model = load_model(folder)
            self.assertEqual(model.vectors.dtype, np.float16)
            self.assertEqual(model.meta["bits"], 16)

    def test_round_does_not_run_the_full_training(self):
        self.train()
        with mock.patch("tune.train_full") as full:
            self.drive()
        full.assert_not_called()

    def test_missing_model_is_train_error(self):
        with self.assertRaisesRegex(TrainError, "train"):
            self.drive()

    def test_no_budget_is_train_error(self):
        self.train()
        with self.assertRaisesRegex(TrainError, "budget"):
            self.drive(budget_hours=0)


class CoverageTest(unittest.TestCase):
    def test_zipf3_coverage(self):
        share = zipf3_coverage(["the", "of", "zzzqqq"])
        self.assertGreater(share, 0.0)
        self.assertLess(share, 0.01)


class CommandTest(unittest.TestCase):
    def test_train_corpus_command(self):
        import cli

        with mock.patch("tune.train_corpus") as train:
            self.assertEqual(
                cli.main(["train", "--corpus", "data/sample_texts", "--out", "x"]), 0
            )
        args = train.call_args.args
        self.assertGreater(len(args[0]), 1)
        self.assertEqual(args[3], "x")

    def test_corpus_with_no_out_keeps_the_saved_model(self):
        import cli

        with mock.patch("tune.train_corpus") as train:
            self.assertEqual(cli.main(["train", "--corpus", "data/sample_texts"]), 0)
        self.assertEqual(train.call_args.args[3], os.path.join(cli.DATA, "smoke"))

    def test_lost_worker_is_one_line(self):
        import cli

        with mock.patch("tune.train_full", side_effect=BrokenProcessPool("gone")):
            with mock.patch("builtins.print") as shown:
                self.assertEqual(cli.main(["train"]), 1)
        self.assertIn("worker", str(shown.call_args))

    def test_train_stall_exits_3(self):
        import cli

        with mock.patch("tune.train_full", return_value={"decision": "stall"}):
            self.assertEqual(cli.main(["train"]), 3)

    def test_train_freeze_exits_0(self):
        import cli

        with mock.patch("tune.train_full", return_value={"decision": "freeze"}):
            self.assertEqual(cli.main(["train"]), 0)

    def test_train_error_is_one_line(self):
        import cli

        with mock.patch("tune.train_full", side_effect=TrainError("no text")):
            with mock.patch("builtins.print") as shown:
                self.assertEqual(cli.main(["train"]), 1)
        self.assertIn("no text", str(shown.call_args))

    def test_measure_only_command(self):
        import cli

        with mock.patch("tune.measure_only") as measure:
            cli.main(["train", "--measure-only", "--words", "1000"])
        self.assertEqual(measure.call_args.args[0], 1000)

    def test_method_round_freeze_exits_0(self):
        import cli

        found = {"decision": "freeze"}
        with mock.patch("tune.run_method_round", return_value=found) as run:
            with mock.patch("tune.train_full") as full:
                self.assertEqual(cli.main(["train", "--method-round"]), 0)
        full.assert_not_called()
        self.assertEqual(run.call_args.args[0], os.path.join(cli.DATA, "work"))

    def test_method_round_stall_exits_3(self):
        import cli

        with mock.patch("tune.run_method_round", return_value={"decision": "stall"}):
            self.assertEqual(cli.main(["train", "--method-round"]), 3)

    def test_method_round_error_is_one_line(self):
        import cli

        with mock.patch("tune.run_method_round", side_effect=TrainError("no model")):
            with mock.patch("builtins.print") as shown:
                self.assertEqual(cli.main(["train", "--method-round"]), 1)
        self.assertIn("no model", str(shown.call_args))


class SettingsSafetyTest(unittest.TestCase):
    def test_directional_skipped_at_twenty_thousand(self):
        big = dataclasses.replace(CountSettings(), n_columns=20000, directional=True)
        self.assertGreater(big.table_columns(), MAX_TABLE_COLUMNS)

    def test_sweep_table_leaves_room_in_the_memory_limit(self):
        self.assertLessEqual(MAX_ROWS * MAX_UINT64_COLUMNS * 8, 8e9)

    def test_widest_table_leaves_room_in_the_memory_limit(self):
        self.assertLessEqual(MAX_ROWS * MAX_TABLE_COLUMNS * 4, 8e9)


if __name__ == "__main__":
    unittest.main()
