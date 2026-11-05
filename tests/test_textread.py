import os
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from textread import (
    Cleaning,
    group_order,
    list_groups,
    read_group,
    read_text_file,
    tokenize_text,
)


class TokenizeTest(unittest.TestCase):
    def test_line_mode_drops_short_lines(self):
        text = "One two three four five\nToo short\nSix seven eight nine ten"
        self.assertEqual(
            tokenize_text(text, Cleaning()),
            [
                ["one", "two", "three", "four", "five"],
                ["six", "seven", "eight", "nine", "ten"],
            ],
        )

    def test_null_and_empty(self):
        self.assertEqual(tokenize_text(None, Cleaning()), [])
        self.assertEqual(tokenize_text("", Cleaning()), [])

    def test_apostrophe_split(self):
        cleaning = Cleaning(min_line_words=0)
        self.assertEqual(tokenize_text("don't", cleaning), [["don", "t"]])

    def test_apostrophe_join(self):
        cleaning = Cleaning(apostrophes="join", min_line_words=0)
        self.assertEqual(tokenize_text("don't don’t", cleaning), [["dont", "dont"]])

    def test_accent_split(self):
        cleaning = Cleaning(min_line_words=0)
        self.assertEqual(tokenize_text("café", cleaning), [["caf"]])

    def test_accent_fold(self):
        cleaning = Cleaning(accents="fold", min_line_words=0)
        self.assertEqual(tokenize_text("café", cleaning), [["cafe"]])

    def test_sentence_mode(self):
        cleaning = Cleaning(breaks="sentence", min_line_words=0)
        self.assertEqual(
            tokenize_text("a b c. d e f", cleaning), [["a", "b", "c"], ["d", "e", "f"]]
        )

    def test_long_token_keeps_its_position(self):
        long_word = "x" * 40
        cleaning = Cleaning(min_line_words=0)
        self.assertEqual(
            tokenize_text(f"a {long_word} b", cleaning), [["a", long_word, "b"]]
        )

    def test_replacement_and_digits_are_breaks_between_words(self):
        cleaning = Cleaning(min_line_words=0)
        self.assertEqual(tokenize_text("a�b 42 c", cleaning), [["a", "b", "c"]])


class GroupOrderTest(unittest.TestCase):
    def test_group_order_interleave(self):
        uk = [("u", i) for i in range(109)]
        fine = [("f", i) for i in range(14874)]
        order = group_order(uk, fine)
        self.assertEqual(sorted(order), sorted(uk + fine))
        positions = [i for i, group in enumerate(order) if group[0] == "u"]
        gaps = [b - a - 1 for a, b in zip(positions, positions[1:])]
        self.assertTrue(all(gap in (136, 137) for gap in gaps))
        self.assertEqual(order, group_order(uk, fine))

    def test_group_order_shuffles(self):
        fine = [("f", i) for i in range(1000)]
        order = group_order([("u", 0)], fine)
        self.assertNotEqual([g for g in order if g[0] == "f"], fine)


class ParquetTest(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(folder.cleanup)
        self.path = os.path.join(folder.name, "a.parquet")
        table = pa.table({"text": ["a b", None, "c"], "url": ["u1", "u2", "u3"]})
        pq.write_table(table, self.path, row_group_size=2)

    def test_list_groups(self):
        self.assertEqual(list_groups([self.path]), [(self.path, 0), (self.path, 1)])

    def test_read_group_with_null_text(self):
        self.assertEqual(read_group(self.path, 0, ("text",))["text"], ["a b", None])

    def test_read_group_columns(self):
        got = read_group(self.path, 1, ("text", "url"))
        self.assertEqual(got, {"text": ["c"], "url": ["u3"]})

    def test_read_text_file_joins_lines(self):
        path = self.path + ".txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write("one\ntwo\r\nthree")
        self.assertEqual(read_text_file(path).split(), ["one", "two", "three"])
        self.assertNotIn("\n", read_text_file(path))


if __name__ == "__main__":
    unittest.main()
