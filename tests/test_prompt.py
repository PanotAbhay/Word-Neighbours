import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from model import save_model
from tests.test_model import VECTORS, WORDS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_cli(args, text=""):
    return subprocess.run(
        [sys.executable, os.path.join(ROOT, "cli.py"), *args],
        input=text,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


class PromptTest(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(folder.cleanup)
        self.folder = os.path.join(folder.name, "model")
        save_model(self.folder, WORDS, VECTORS, {})

    def prompt(self, text):
        result = run_cli(["--model", self.folder], text)
        self.assertNotIn("Traceback", result.stderr)
        return result

    def test_capitals_and_punctuation(self):
        out = self.prompt("Apple, PEAR plum chair\nq\n").stdout
        self.assertIn("odd one out: chair", out)

    def test_unknown_listed(self):
        out = self.prompt("apple zzz\nq\n").stdout
        self.assertIn("not in the vocabulary: zzz", out)

    def test_two_words_message(self):
        out = self.prompt("apple pear\nq\n").stdout
        self.assertIn("at least three", out)
        self.assertNotIn("odd one out", out)

    def test_repeat_word_message(self):
        out = self.prompt("apple apple pear chair\nq\n").stdout
        self.assertIn("repeat", out)
        self.assertNotIn("odd one out", out)

    def test_one_word_neighbours(self):
        out = self.prompt("apple\nq\n").stdout
        self.assertIn("pear", out)

    def test_only_punctuation_keeps_prompt_open(self):
        out = self.prompt("?!\napple pear plum chair\nq\n").stdout
        self.assertIn("odd one out: chair", out)

    def test_end_of_input_exits_cleanly(self):
        self.assertEqual(self.prompt("").returncode, 0)

    def test_missing_model_message(self):
        result = run_cli(["--model", os.path.join(self.folder, "nothing")])
        self.assertEqual(result.returncode, 1)
        self.assertIn("python cli.py train", result.stdout)
        self.assertNotIn("Traceback", result.stderr)

    def test_corrupt_model_message(self):
        path = os.path.join(self.folder, "vectors.npy")
        with open(path, "rb") as f:
            data = f.read()
        with open(path, "wb") as f:
            f.write(data[:60])
        result = run_cli(["--model", self.folder])
        self.assertEqual(result.returncode, 1)
        self.assertNotIn("Traceback", result.stderr)

    def test_prompt_needs_numpy_only(self):
        code = (
            "import sys, cli; "
            "assert not {'pyarrow', 'scipy', 'wordfreq'} & set(sys.modules)"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_load_puzzles_reads_two_fields_of_tagged_line(self):
        import cli

        path = os.path.join(self.folder, "p.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("# note\napple pear plum chair ; chair ; part=noun type=close\n")
        self.assertEqual(
            cli.load_puzzles(path), [(["apple", "pear", "plum", "chair"], "chair")]
        )


class CommandTest(unittest.TestCase):
    def test_download_lists_only(self):
        import cli

        with mock.patch("fetch.download_all") as download:
            self.assertEqual(cli.main(["download", "--lists"]), 0)
        args, kwargs = download.call_args
        self.assertEqual(args[1], os.path.join(ROOT, "data", "raw"))
        self.assertEqual(kwargs["kinds"], ("lists",))

    def test_download_everything(self):
        import cli

        with mock.patch("fetch.download_all") as download:
            self.assertEqual(cli.main(["download"]), 0)
        self.assertEqual(download.call_args.kwargs["kinds"], ("lists", "text"))

    def test_download_failure_is_one_line(self):
        import cli
        from fetch import DownloadError

        with mock.patch("fetch.download_all", side_effect=DownloadError("gone")):
            with mock.patch("builtins.print") as shown:
                self.assertEqual(cli.main(["download"]), 1)
        self.assertIn("gone", str(shown.call_args))

    def test_puzzles_passes_seed(self):
        import cli

        with mock.patch("puzzlebuild.build_all") as build:
            self.assertEqual(cli.main(["puzzles", "--seed", "3"]), 0)
        args, kwargs = build.call_args
        self.assertEqual(args[0], os.path.join(ROOT, "data", "raw", "lists"))
        self.assertEqual(args[1], os.path.join(ROOT, "data"))
        self.assertEqual(kwargs["seed"], 3)

    def test_puzzles_shortfall_is_one_line(self):
        import cli

        short = ValueError("only 8 of 16 close puzzles for ship")
        with mock.patch("puzzlebuild.build_all", side_effect=short):
            with mock.patch("builtins.print") as shown:
                self.assertEqual(cli.main(["puzzles", "--seed", "3"]), 1)
        self.assertIn("seed 3", str(shown.call_args))
        self.assertIn("for ship", str(shown.call_args))

    def test_puzzles_default_seed(self):
        import cli

        with mock.patch("puzzlebuild.build_all") as build:
            cli.main(["puzzles"])
        self.assertEqual(build.call_args.kwargs["seed"], 780)


if __name__ == "__main__":
    unittest.main()
