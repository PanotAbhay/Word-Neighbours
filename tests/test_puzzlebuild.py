import collections
import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile

import fileio
from puzzlebuild import (
    build_all,
    build_classes,
    load_twins,
    parse_varcon,
    parse_voc,
    split_words,
    zipf_of,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LISTS = os.path.join(ROOT, "data", "raw", "lists")
HAVE_LISTS = os.path.exists(os.path.join(LISTS, "voc.tab"))
Line = collections.namedtuple("Line", "words answer tags")


def read_lines(path):
    lines = []
    for text in fileio.read_lines(path):
        words, answer, tags = text.split(" ; ")
        tags = dict(tag.split("=") for tag in tags.split())
        lines.append(Line(words.split(), answer, tags))
    return lines


def words_of(lines):
    return {word for line in lines for word in line.words}


def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class ParseTest(unittest.TestCase):
    def test_varcon_exact_case(self):
        text = (
            "A Cv: gray / AV B C: grey\nA: Acer / B: Acre\nA: airplane / B: aeroplane\n"
        )
        self.assertEqual(
            parse_varcon(text), [("gray", "grey"), ("airplane", "aeroplane")]
        )

    def test_varcon_strips_notes(self):
        text = "A: absinthe / AV B: absinth | :1\nA: armories / B: armouries # note\n"
        self.assertEqual(
            parse_varcon(text), [("absinthe", "absinth"), ("armories", "armouries")]
        )

    def test_voc_first_uk_word_and_no_notes(self):
        text = (
            "truck\tlorry,van\ttruck\n"
            "one-way\tsingle\tone-way\t(ticket)\n"
            "baggage's\tluggage's\tluggage's\n"
        )
        self.assertEqual(parse_voc(text, lambda w: 5.0), [("truck", "lorry")])

    def test_voc_each_single_us_word(self):
        text = "ashcan,garbage can,trashcan\tdustbin\tgarbage can\n"
        self.assertEqual(
            parse_voc(text, lambda w: 5.0),
            [("ashcan", "dustbin"), ("trashcan", "dustbin")],
        )

    def test_voc_zipf_floor_needs_both_sides(self):
        text = "truck\tlorry\ttruck\n"
        self.assertEqual(parse_voc(text, lambda w: 2.0), [])
        floor = lambda w: 5.0 if w == "truck" else 2.0
        self.assertEqual(parse_voc(text, floor), [])


class SplitTest(unittest.TestCase):
    def fake_classes(self):
        words = {
            "aa": ["a1", "a2", "a3", "a4", "a5", "a6"],
            "bb": ["b1", "b2", "b3", "b4", "b5", "b6", "a1"],
            "vt": ["pin1", "pin2", "pin3"],
            "vn": ["net1", "net2", "net3"],
        }
        return build_fake(words)

    def test_split_balances_each_class(self):
        split = split_words(self.fake_classes(), {}, 780)
        for name in ("aa", "bb"):
            count = collections.Counter(
                split[w] for w in self.fake_classes().words[name]
            )
            self.assertLessEqual(abs(count[0] - count[1]), 1)

    def test_split_pins_verbs(self):
        split = split_words(self.fake_classes(), {}, 780)
        self.assertEqual({split[w] for w in ("pin1", "pin2", "pin3")}, {1})
        self.assertEqual({split[w] for w in ("net1", "net2", "net3")}, {0})

    def test_twins_stay_together(self):
        twins = {"a1": "z1", "z1": "a1", "pin1": "z2", "z2": "pin1"}
        split = split_words(self.fake_classes(), twins, 780)
        self.assertEqual(split["a1"], split["z1"])
        self.assertEqual(split["z2"], 1)

    def test_split_follows_seed(self):
        self.assertNotEqual(
            split_words(self.fake_classes(), {}, 780),
            split_words(self.fake_classes(), {}, 781),
        )


def build_fake(words):
    from puzzlebuild import Classes

    member = {}
    for name, listed in words.items():
        for word in listed:
            member.setdefault(word, set()).add(name)
    verb_side = {"vt": 1, "vn": 0}
    return Classes(
        words,
        member,
        {name: name for name in words},
        {name: "verb" if name in verb_side else "noun" for name in words},
        verb_side,
    )


@unittest.skipUnless(HAVE_LISTS, "run: python cli.py download --lists")
class RealListsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.classes = build_classes(LISTS, zipf_of)
        folder = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.addClassCleanup(folder.cleanup)
        cls.folder = folder.name
        cls.counts = build_all(LISTS, cls.folder)
        cls.tune = read_lines(os.path.join(cls.folder, "puzzles_tune.txt"))
        cls.test = read_lines(os.path.join(cls.folder, "puzzles_test.txt"))

    def test_varcon_full_count(self):
        text = open(os.path.join(LISTS, "varcon.txt"), encoding="latin-1").read()
        self.assertEqual(len(parse_varcon(text)), 12718)

    def test_adjective_groups_exact(self):
        self.assertIn("fat", self.classes.words["body weight"])
        self.assertIn("dead", self.classes.words["aliveness animation"])
        self.assertIn("big", self.classes.words["size"])
        self.assertEqual(self.classes.part["size"], "adjective")
        self.assertEqual(self.classes.part["colour"], "adjective")

    def test_first_twelve_rule(self):
        self.assertEqual(self.classes.words["fruit"][:3], ["apple", "orange", "pear"])
        self.assertNotIn("flag", self.classes.words["flower"])
        self.assertIn("flower", self.classes.member["flag"])

    def test_verbnet_excludes_essli(self):
        essli = set()
        for name, side in self.classes.verb_side.items():
            if side == 1:
                essli |= set(self.classes.words[name])
        self.assertEqual(len(essli), 45)
        for name, side in self.classes.verb_side.items():
            if side == 0:
                self.assertEqual(set(self.classes.words[name]) & essli, set())
                self.assertLessEqual(len(self.classes.words[name]), 8)
        self.assertIn("hit", self.classes.words["hit-18.1"])

    def test_no_word_shared(self):
        self.assertEqual(words_of(self.tune) & words_of(self.test), set())

    def test_each_puzzle_is_well_formed(self):
        for line in self.tune + self.test:
            self.assertEqual(len(set(line.words)), 4)
            self.assertIn(line.answer, line.words)
            target = line.tags["target"].replace("_", " ")
            self.assertNotIn(target, self.classes.member.get(line.answer, set()))
            triple = [w for w in line.words if w != line.answer]
            odd = line.tags["odd"].replace("_", " ")
            for word in triple:
                self.assertNotIn(odd, self.classes.member.get(word, set()))

    def test_base_sizes(self):
        for lines in (self.tune, self.test):
            sizes = collections.Counter(
                (line.tags["part"], line.tags["type"])
                for line in lines
                if line.tags["form"] != "uk"
            )
            self.assertEqual(
                sizes,
                {
                    ("noun", "close"): 800,
                    ("noun", "unrelated"): 800,
                    ("adjective", "close"): 100,
                    ("adjective", "unrelated"): 100,
                    ("verb", "close"): 100,
                    ("verb", "unrelated"): 100,
                },
            )

    def test_class_balance_within_type(self):
        for lines in (self.tune, self.test):
            per_class = collections.defaultdict(collections.Counter)
            for line in lines:
                if line.tags["form"] != "uk":
                    key = (line.tags["part"], line.tags["type"])
                    per_class[key][line.tags["target"]] += 1
            for counts in per_class.values():
                self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_unrelated_odd_is_from_another_domain(self):
        for line in self.tune + self.test:
            if line.tags["type"] == "unrelated":
                self.assertEqual(
                    self.classes.part[line.tags["odd"].replace("_", " ")], "noun"
                )
                self.assertNotEqual(
                    self.classes.domain[line.tags["odd"].replace("_", " ")],
                    line.tags["domain"],
                )

    def test_twin_pairs_match(self):
        for lines in (self.tune, self.test):
            pairs = collections.defaultdict(list)
            for line in lines:
                if line.tags["pair"] != "-":
                    pairs[line.tags["pair"]].append(line)
                else:
                    self.assertEqual(line.tags["form"], "base")
            self.assertGreater(len(pairs), 0)
            for found in pairs.values():
                self.assertEqual(sorted(p.tags["form"] for p in found), ["uk", "us"])
                self.assertNotEqual(found[0].words, found[1].words)

    def test_known_twin_forms(self):
        twins = load_twins(LISTS, zipf_of)
        self.assertEqual(twins.uk_form["gray"], "grey")
        self.assertEqual(twins.us_form["grey"], "gray")
        self.assertEqual(twins.uk_form["truck"], "lorry")

    def test_verbs_on_the_right_side(self):
        for name, side in self.classes.verb_side.items():
            other = self.tune if side == 1 else self.test
            shared = set(self.classes.words[name]) & words_of(other)
            self.assertEqual(shared, set(), name)

    def test_same_seed_same_files_any_hash_seed(self):
        hashes = []
        for hash_seed in ("1", "2"):
            folder = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
            self.addCleanup(folder.cleanup)
            code = (
                "import sys; from puzzlebuild import build_all; "
                "build_all(sys.argv[1], sys.argv[2])"
            )
            subprocess.run(
                [sys.executable, "-c", code, LISTS, folder.name],
                cwd=ROOT,
                check=True,
                capture_output=True,
                env={**os.environ, "PYTHONHASHSEED": hash_seed},
            )
            hashes.append(
                [
                    file_hash(os.path.join(folder.name, name))
                    for name in ("puzzles_tune.txt", "puzzles_test.txt")
                ]
            )
        self.assertEqual(hashes[0], hashes[1])
        self.assertEqual(
            hashes[0][0], file_hash(os.path.join(self.folder, "puzzles_tune.txt"))
        )

    def test_other_seed_differs(self):
        folder = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(folder.cleanup)
        build_all(LISTS, folder.name, seed=782)
        self.assertNotEqual(
            file_hash(os.path.join(folder.name, "puzzles_test.txt")),
            file_hash(os.path.join(self.folder, "puzzles_test.txt")),
        )

    def test_reports_counts(self):
        self.assertEqual(set(self.counts), {"tune", "test", "classes", "dropped"})
        self.assertIn("fruit", self.counts["classes"])

    def test_verbnet_zip_is_only_data(self):
        with zipfile.ZipFile(os.path.join(LISTS, "verbnet3.zip")) as z:
            self.assertIn("verbnet3/hit-18.1.xml", z.namelist())


if __name__ == "__main__":
    unittest.main()
