import os
import tempfile
import unittest

import fileio
from model import load_model, save_model
from scoring import (
    TestRefused,
    class_bootstrap,
    format_report,
    load_probes,
    load_tagged_puzzles,
    model_hash,
    puzzle_results,
    run_test,
    score_report,
    wilson,
)
from tests.test_model import VECTORS, WORDS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def line(
    words,
    answer,
    part="noun",
    kind="close",
    target="fruit",
    domain="food",
    form="base",
    pair="-",
):
    return (
        f"{words} ; {answer} ; part={part} type={kind} target={target} odd=x "
        f"domain={domain} form={form} pair={pair}"
    )


LINES = [
    line("apple pear plum chair", "chair"),
    line("apple pear plum zzz", "zzz", kind="unrelated"),
    line("chair sofa apple pear", "pear", part="verb", target="seat", domain="verb"),
    line(
        "apple pear plum chair",
        "chair",
        part="adjective",
        target="tree",
        domain="adjective",
    ),
    line("apple pear plum chair", "chair", form="us", pair="p0001"),
    line("apple pear plum sofa", "plum", form="uk", pair="p0001"),
]


class Fixture(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(folder.cleanup)
        self.folder = folder.name
        self.model_dir = os.path.join(self.folder, "model")
        self.puzzles = os.path.join(self.folder, "puzzles_test.txt")
        self.log = os.path.join(self.folder, "test_log.txt")
        fileio.write_lines(self.puzzles, LINES)
        self.save_model({"frozen": True})

    def save_model(self, meta, vectors=VECTORS, folder=None):
        save_model(folder or self.model_dir, WORDS, vectors, meta)

    def run_it(self, **kwargs):
        return run_test(self.model_dir, self.puzzles, self.log, **kwargs)


class RangesTest(unittest.TestCase):
    def test_wilson_at_0_90_and_2000(self):
        lo, hi = wilson(1800, 2000)
        self.assertAlmostEqual(lo, 0.886, places=2)
        self.assertAlmostEqual(hi, 0.912, places=2)

    def test_wilson_empty(self):
        self.assertEqual(wilson(0, 0), (0.0, 0.0))

    def test_bootstrap_wider_when_classes_differ(self):
        targets = [f"c{i}" for i in range(10) for _ in range(10)]
        even = [j % 2 == 0 for i in range(10) for j in range(10)]
        split = [i < 5 for i in range(10) for j in range(10)]
        lo_even, hi_even = class_bootstrap(even, targets)
        lo_split, hi_split = class_bootstrap(split, targets)
        self.assertGreater(hi_split - lo_split, hi_even - lo_even)

    def test_bootstrap_is_deterministic(self):
        targets = ["a", "a", "b", "c"]
        results = [True, False, True, False]
        self.assertEqual(
            class_bootstrap(results, targets), class_bootstrap(results, targets)
        )


class ReportTest(Fixture):
    def report(self):
        puzzles = load_tagged_puzzles(self.puzzles)
        return score_report(puzzles, load_model(self.model_dir))

    def test_load_tagged_puzzles(self):
        first = load_tagged_puzzles(self.puzzles)[0]
        self.assertEqual(first.words, ["apple", "pear", "plum", "chair"])
        self.assertEqual(first.answer, "chair")
        self.assertEqual(first.tags["part"], "noun")

    def test_report_groups_and_unknown_is_wrong(self):
        report = self.report()
        self.assertEqual(report["overall"]["n"], 6)
        self.assertEqual(report["unknown"]["n"], 1)
        self.assertEqual(report["unknown"]["score"], 0.0)
        self.assertEqual(report["by_part"]["verb"]["n"], 1)
        self.assertEqual(report["by_type"]["unrelated"]["score"], 0.0)
        self.assertEqual(report["by_domain"]["food"]["n"], 4)
        self.assertAlmostEqual(report["overall"]["score"], 3 / 6)
        for field in ("score", "n", "binomial", "bootstrap"):
            self.assertIn(field, report["overall"])

    def test_puzzle_results_lists_one_answer_per_puzzle(self):
        puzzles = load_tagged_puzzles(self.puzzles)
        got = puzzle_results(puzzles, load_model(self.model_dir))
        self.assertEqual(got, [True, False, False, True, True, False])

    def test_twins_scored_per_form(self):
        report = self.report()
        self.assertEqual(report["twins"]["n_pairs"], 1)
        self.assertEqual(report["twins"]["us"]["score"], 1.0)
        self.assertEqual(report["twins"]["uk"]["score"], 0.0)

    def test_format_shows_parts_and_fixed_lines(self):
        report = self.report()
        report["probes"] = {"modern": {"score": 0.5, "n": 10}}
        text = format_report(report)
        for piece in (
            "verb",
            "adjective",
            "modern",
            "This test uses about 450 words of 28,338 at Zipf 3",
            "Pass rule: pooled, nouns are 80% of the test",
            "Modern words",
            "UK/US vocabulary",
            "45 test verbs",
            "Battig",
        ):
            self.assertIn(piece, text)

    def test_format_names_a_miss(self):
        self.assertIn("MISS", format_report(self.report()))


class GuardsTest(Fixture):
    def test_refuses_unfrozen_model(self):
        self.save_model({"frozen": False}, folder=self.model_dir + "_a")
        with self.assertRaises(TestRefused):
            run_test(self.model_dir + "_a", self.puzzles, self.log)

    def test_refuses_model_with_no_frozen_field(self):
        self.save_model({}, folder=self.model_dir + "_a")
        with self.assertRaises(TestRefused):
            run_test(self.model_dir + "_a", self.puzzles, self.log)

    def test_refuses_stalled_model_without_approval(self):
        self.save_model({"frozen": True, "stalled": True}, folder=self.model_dir + "_a")
        with self.assertRaises(TestRefused):
            run_test(self.model_dir + "_a", self.puzzles, self.log)
        self.assertFalse(os.path.exists(self.log))

    def test_stalled_model_runs_with_approval(self):
        self.save_model({"frozen": True, "stalled": True}, folder=self.model_dir + "_a")
        report = run_test(
            self.model_dir + "_a", self.puzzles, self.log, owner_approved=True
        )
        self.assertEqual(report["overall"]["n"], 6)

    def test_refuses_new_model_on_same_test(self):
        self.run_it()
        other = self.model_dir + "_other"
        self.save_model({"frozen": True}, VECTORS[::-1].copy(), folder=other)
        with self.assertRaises(TestRefused):
            run_test(other, self.puzzles, self.log)

    def test_new_model_on_new_test_runs(self):
        self.run_it()
        other = self.model_dir + "_other"
        self.save_model({"frozen": True}, VECTORS[::-1].copy(), folder=other)
        fresh = os.path.join(self.folder, "fresh.txt")
        fileio.write_lines(fresh, LINES[:3])
        self.assertEqual(run_test(other, fresh, self.log)["overall"]["n"], 3)

    def test_same_model_rerun_is_logged(self):
        self.run_it()
        self.run_it()
        self.assertEqual(len(fileio.read_lines(self.log)), 2)

    def test_log_line_fields(self):
        self.run_it()
        text = fileio.read_lines(self.log)[0]
        self.assertIn(f"model={model_hash(self.model_dir)}", text)
        self.assertRegex(text, r"^\d{4}-\d{2}-\d{2} ")
        self.assertRegex(text, r"puzzles=[0-9a-f]{64}")
        self.assertRegex(text, r"score=0\.5000")

    def test_model_hash_uses_vectors_and_words(self):
        first = model_hash(self.model_dir)
        self.save_model(
            {"frozen": True}, VECTORS[::-1].copy(), folder=self.model_dir + "_b"
        )
        self.assertNotEqual(first, model_hash(self.model_dir + "_b"))

    def test_report_holds_probe_scores(self):
        report = self.run_it()
        self.assertEqual(set(report["probes"]), {"easy", "near", "modern", "ukus"})
        self.assertEqual(report["probes"]["modern"]["n"], 10)


class ProbesTest(unittest.TestCase):
    def test_probe_tier_sizes(self):
        probes = load_probes(os.path.join(ROOT, "data", "probes.txt"))
        sizes = {tier: len(items) for tier, items in probes.items()}
        self.assertEqual(sizes, {"easy": 30, "near": 20, "modern": 10, "ukus": 15})
        self.assertEqual(
            probes["easy"][0], (["apple", "pear", "plum", "chair"], "chair")
        )

    @unittest.skipUnless(os.path.exists(os.path.join(ROOT, "tune.py")), "needs Task 8")
    def test_tune_module_has_no_test_path(self):
        self.assertNotIn("puzzles_test", fileio.read_text(os.path.join(ROOT, "tune.py")))


class CommandTest(Fixture):
    def test_test_command_refusal_is_one_line(self):
        from unittest import mock

        import cli

        with mock.patch("scoring.run_test", side_effect=TestRefused("not frozen")):
            with mock.patch("builtins.print") as shown:
                self.assertEqual(cli.main(["test"]), 1)
        self.assertIn("not frozen", str(shown.call_args))

    def test_test_command_passes_approval(self):
        from unittest import mock

        import cli

        report = score_report(
            load_tagged_puzzles(self.puzzles), load_model(self.model_dir)
        )
        with mock.patch("scoring.run_test", return_value=report) as run:
            with mock.patch("builtins.print"):
                self.assertEqual(cli.main(["test", "--owner-approved"]), 0)
        self.assertTrue(run.call_args.kwargs["owner_approved"])


if __name__ == "__main__":
    unittest.main()
