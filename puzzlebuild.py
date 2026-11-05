import csv
import functools
import gzip
import os
import random
import re
import xml.etree.ElementTree as ET
import zipfile
from typing import NamedTuple

import fileio

B, S, P = "battig", "bless", "ap"
NOUN = {
    "fruit": ("food", [(B, "fruit"), (S, "fruit")]),
    "vegetable": ("food", [(B, "vegetable"), (S, "vegetable")]),
    "flavouring": ("food", [(B, "substance to flavor food")]),
    "soft drink": ("food", [(B, "nonalcoholic beverage")]),
    "alcoholic drink": ("food", [(B, "alcoholic beverage")]),
    "dish": ("food", [("oewn", ("dish", "noun.food"))]),
    "bread": ("food", [("oewn", ("bread", "noun.food"))]),
    "meal": ("food", [("oewn", ("meal", "noun.food"))]),
    "mammal": ("animals", [(B, "four-footed animal"), (S, "ground_mammal")]),
    "bird": ("animals", [(B, "bird"), (S, "bird")]),
    "fish": ("animals", [(B, "fish"), (S, "water_animal")]),
    "insect": ("animals", [(B, "insect"), (S, "insect")]),
    "tree": ("plants", [(B, "tree"), (S, "tree")]),
    "flower": ("plants", [(B, "flower")]),
    "clothing": ("clothing", [(B, "article of clothing"), (S, "clothing")]),
    "footwear": ("clothing", [(B, "type of footgear")]),
    "cloth": ("clothing", [(B, "kind of cloth")]),
    "weapon": ("tools", [(B, "weapon"), (S, "weapon")]),
    "tool": ("tools", [(B, "carpenters tool"), (S, "tool")]),
    "utensil": ("tools", [(B, "kitchen utensil")]),
    "metal": ("materials", [(B, "metal")]),
    "precious stone": ("materials", [(B, "precious stone")]),
    "element": ("materials", [(B, "chemical element")]),
    "fuel": ("materials", [(B, "type of fuel")]),
    "time unit": ("measures", [(B, "unit of time")]),
    "distance unit": ("measures", [(B, "unit of distance")]),
    "money": ("measures", [(B, "kind of money")]),
    "building part": ("home", [(B, "part of a building")]),
    "furniture": ("home", [(B, "article of furniture"), (S, "furniture")]),
    "dwelling": ("buildings", [(B, "type of human dwelling")]),
    "religious building": ("buildings", [(B, "bldg for religious servic")]),
    "public building": ("buildings", [(S, "building")]),
    "shop": ("buildings", [("oewn", ("shop", "noun.artifact"))]),
    "vehicle": ("transport", [(B, "type of vehicle"), (S, "vehicle")]),
    "ship": ("transport", [(B, "type of ship")]),
    "music type": ("music", [(B, "type of music")]),
    "instrument": ("music", [(B, "musical instrument"), (S, "musical_instrument")]),
    "occupation": ("people", [(B, "occupation or profession")]),
    "relative": ("people", [(B, "relative")]),
    "clergy": ("people", [(B, "member of the clergy")]),
    "military title": ("people", [(B, "military title")]),
    "body part": ("body", [(B, "part of the human body")]),
    "disease": ("body", [(B, "disease")]),
    "pain": ("body", [(P, "pain")]),
    "weather": ("nature", [(B, "weather phenomenon")]),
    "earth formation": ("nature", [(B, "natural earth formation")]),
    "city": ("places", [(B, "city")]),
    "country": ("places", [(B, "country")]),
    "district": ("places", [(P, "district")]),
    "appliance": (
        "devices",
        [(S, "appliance"), ("oewn", ("home appliance", "noun.artifact"))],
    ),
    "electronics": (
        "devices",
        [
            ("oewn", ("electronic equipment", "noun.artifact")),
            ("oewn", ("computer", "noun.artifact")),
        ],
    ),
    "sport": ("alone1", [(B, "sport")]),
    "reading": ("alone2", [(B, "type of reading material")]),
    "crime": ("alone3", [(B, "crime")]),
    "science": ("alone4", [(B, "science")]),
    "feeling": ("alone5", [(P, "feeling")]),
    "occasion": ("alone6", [(P, "social_occasion")]),
    "social unit": ("alone7", [(P, "social_unit")]),
}
ADJECTIVE_GROUPS = [
    "size",
    "age",
    "temperature",
    "wetness",
    "duration length",
    "strength",
    "beauty",
    "cleanness",
    "fastness speed",
    "breadth width",
    "body weight",
    "quality",
    "fear fearfulness",
    "aliveness animation",
    "difficultness difficulty",
]
VERBNET = [
    "assuming_position-50",
    "begin-55.1",
    "break-45.1",
    "build-26.1",
    "comprehend-87.2",
    "cooking-45.3",
    "get-13.5.1",
    "give-13.1",
    "hit-18.1",
    "multiply-108",
    "murder-42.1",
    "nonverbal_expression-40.2",
    "push-12",
    "run-51.3.2",
    "say-37.7",
    "see-30.1",
    "weather-57",
]
SIZES = {"noun": 800, "adjective": 100, "verb": 100}
WORD = re.compile("[a-z]+")
HALF_FILES = ("puzzles_tune.txt", "puzzles_test.txt")


class Classes(NamedTuple):
    words: dict
    member: dict
    domain: dict
    part: dict
    verb_side: dict
    dropped: tuple = ()


class Twins(NamedTuple):
    other: dict
    us_form: dict
    uk_form: dict
    vocabulary: frozenset = frozenset()


class Oewn(NamedTuple):
    senses: dict
    lexfile: dict
    members: dict
    relations: dict


@functools.lru_cache(maxsize=None)
def zipf_of(word):
    from wordfreq import zipf_frequency

    return zipf_frequency(word, "en", wordlist="large")


def is_word(word):
    return WORD.fullmatch(word) is not None


def by_zipf(words, zipf):
    return sorted(words, key=lambda word: (-zipf(word), word))


def parse_varcon(text):
    pairs = []
    for line in text.split("\n"):
        line = line.split("#")[0].split("|")[0].strip()
        us, uk = [], []
        for part in line.split("/"):
            if ":" not in part:
                continue
            tags, word = part.split(":", 1)
            tags = tags.split()
            word = word.strip()
            if not is_word(word):
                continue
            if "A" in tags and "B" not in tags:
                us.append(word)
            if "B" in tags and "A" not in tags:
                uk.append(word)
        if us and uk:
            pairs.append((us[0], uk[0]))
    return pairs


def parse_voc(text, zipf):
    pairs = []
    for line in text.split("\n"):
        fields = line.split("\t")
        if len(fields) != 3 or "'" in line:
            continue
        uk = fields[1].split(",")[0].strip()
        if not is_word(uk) or zipf(uk) < 3:
            continue
        for us in fields[0].split(","):
            us = us.strip()
            if is_word(us) and zipf(us) >= 3:
                pairs.append((us, uk))
    return pairs


def load_twins(list_dir, zipf):
    with open(os.path.join(list_dir, "varcon.txt"), encoding="latin-1") as f:
        spelling = parse_varcon(f.read())
    vocabulary = parse_voc(fileio.read_text(os.path.join(list_dir, "voc.tab")), zipf)
    other, us_form, uk_form = {}, {}, {}
    from_vocabulary = set()
    for pairs, is_vocabulary in ((spelling, False), (vocabulary, True)):
        for us, uk in pairs:
            if us == uk or us in other or uk in other:
                continue
            other[us], other[uk] = uk, us
            us_form[us], us_form[uk] = us, us
            uk_form[us], uk_form[uk] = uk, uk
            if is_vocabulary:
                from_vocabulary.update((us, uk))
    return Twins(other, us_form, uk_form, frozenset(from_vocabulary))


def read_csv(list_dir, name):
    with open(os.path.join(list_dir, name), encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    categories = {}
    for _, category, word in rows[1:]:
        categories.setdefault(category, []).append(word.strip().lower())
    return categories


@functools.lru_cache(maxsize=None)
def load_oewn(path):
    with gzip.open(path) as f:
        lexicon = ET.fromstring(f.read())[0]
    senses, entry_word = {}, {}
    lexfile, member_ids, relations = {}, {}, {}
    for item in lexicon:
        if item.tag == "LexicalEntry":
            lemma = item.find("Lemma")
            word = lemma.get("writtenForm")
            pos = lemma.get("partOfSpeech").replace("s", "a")
            entry_word[item.get("id")] = word
            found = senses.setdefault(word, {}).setdefault(pos, [])
            found.extend(sense.get("synset") for sense in item.findall("Sense"))
        elif item.tag == "Synset":
            name = item.get("id")
            lexfile[name] = item.get("lexfile")
            member_ids[name] = item.get("members", "").split()
            links = relations.setdefault(name, {})
            for link in item.findall("SynsetRelation"):
                links.setdefault(link.get("relType"), []).append(link.get("target"))
    members = {
        name: sorted(entry_word[entry] for entry in ids if entry in entry_word)
        for name, ids in member_ids.items()
    }
    return Oewn(senses, lexfile, members, relations)


def oewn_closure(oewn, lemma, lexfile):
    for synset in oewn.senses[lemma]["n"]:
        if oewn.lexfile[synset] == lexfile:
            break
    else:
        raise ValueError(f"no {lexfile} noun sense for {lemma}")
    closure, waiting = set(), [synset]
    while waiting:
        synset = waiting.pop()
        if synset not in closure:
            closure.add(synset)
            waiting.extend(oewn.relations[synset].get("hyponym", []))
    return closure


def oewn_noun_words(oewn, closure, zipf):
    listed, kept = [], []
    for word in sorted(oewn.senses):
        senses = oewn.senses[word]
        if is_word(word) and "n" in senses and senses["n"][0] in closure:
            listed.append(word)
            others = [len(found) for pos, found in senses.items() if pos != "n"]
            if len(senses["n"]) >= max(others, default=0) and zipf(word) >= 3:
                kept.append(word)
    return listed, by_zipf(kept, zipf)[:12]


def group_key(oewn, noun_synset):
    return " ".join(oewn.members[noun_synset][:2])


def first_sense_groups(oewn, synset):
    links = oewn.relations[synset]
    nouns = list(links.get("attribute", []))
    for near in links.get("similar", []):
        nouns.extend(oewn.relations[near].get("attribute", []))
    return {group_key(oewn, noun) for noun in nouns}


def adjective_groups(oewn, zipf):
    listed = {name: [] for name in ADJECTIVE_GROUPS}
    kept = {name: [] for name in ADJECTIVE_GROUPS}
    for word in sorted(oewn.senses):
        senses = oewn.senses[word]
        if not is_word(word) or "a" not in senses:
            continue
        groups = first_sense_groups(oewn, senses["a"][0])
        for name in ADJECTIVE_GROUPS:
            if name in groups:
                listed[name].append(word)
                wins = all(
                    len(senses["a"]) >= len(senses.get(pos, [])) for pos in ("n", "v")
                )
                if wins and zipf(word) >= 3.5:
                    kept[name].append(word)
    return listed, {name: by_zipf(kept[name], zipf)[:8] for name in kept}


def verbnet_classes(path, oewn, essli, zipf):
    listed, kept = {}, {}
    with zipfile.ZipFile(path) as z:
        for name in VERBNET:
            root = ET.fromstring(z.read(f"verbnet3/{name}.xml"))
            words = sorted({m.get("name").lower() for m in root.iter("MEMBER")})
            good = []
            for word in words:
                senses = oewn.senses.get(word, {})
                verbs = len(senses.get("v", []))
                if (
                    is_word(word)
                    and word not in essli
                    and zipf(word) >= 3
                    and verbs > len(senses.get("n", []))
                    and verbs >= len(senses.get("a", []))
                ):
                    good.append(word)
            listed[name] = words
            kept[name] = by_zipf(good, zipf)[:8]
    return listed, kept


def build_classes(list_dir, zipf):
    tables = {
        B: read_csv(list_dir, "battig.csv"),
        S: read_csv(list_dir, "bless.csv"),
        P: read_csv(list_dir, "ap.csv"),
    }
    oewn = load_oewn(os.path.join(list_dir, "english-wordnet-2025.xml.gz"))
    words, member, domain, part, verb_side = {}, {}, {}, {}, {}
    dropped = []

    def add(name, listed, kept, in_domain, in_part, size=6):
        domain[name] = in_domain
        part[name] = in_part
        for word in listed:
            member.setdefault(word, set()).add(name)
        if len(kept) >= size:
            words[name] = kept
        else:
            dropped.append(name)

    for name, (in_domain, sources) in NOUN.items():
        listed, kept = [], []
        for kind, key in sources:
            if kind == "oewn":
                closure = oewn_closure(oewn, key[0], key[1])
                full, good = oewn_noun_words(oewn, closure, zipf)
            else:
                full = tables[kind][key]
                good = [w for w in full if is_word(w) and zipf(w) >= 3]
                good = good[:12] if kind == B else good
            listed.extend(full)
            kept.extend(word for word in good if word not in kept)
        add(name, listed, kept, in_domain, "noun")

    colours = tables[B]["color"]
    good = [word for word in colours if is_word(word) and zipf(word) >= 3.5]
    add("colour", colours, good[:12], "adjective", "adjective")
    listed, kept = adjective_groups(oewn, zipf)
    for name in ADJECTIVE_GROUPS:
        add(name, listed[name], kept[name], "adjective", "adjective")

    essli = read_csv(list_dir, "essli-2008.csv")
    for name, verbs in essli.items():
        add(name, verbs, verbs, "verb", "verb", size=3)
        if name in words:
            verb_side[name] = 1
    all_essli = {word for verbs in essli.values() for word in verbs}
    path = os.path.join(list_dir, "verbnet3.zip")
    listed, kept = verbnet_classes(path, oewn, all_essli, zipf)
    for raw in VERBNET:
        name = raw.replace("_", " ")
        add(name, listed[raw], kept[raw], "verb", "verb")
        if name in words:
            verb_side[name] = 0
    return Classes(words, member, domain, part, verb_side, tuple(dropped))


def split_words(classes, twins, seed):
    rng = random.Random(seed)
    split = {}

    def assign(word, half):
        if word not in split:
            split[word] = half
            if word in twins:
                split.setdefault(twins[word], half)

    for side in (1, 0):
        for name in sorted(classes.verb_side):
            if classes.verb_side[name] == side:
                for word in classes.words[name]:
                    assign(word, side)
    for name in sorted(classes.words):
        shuffled = list(classes.words[name])
        rng.shuffle(shuffled)
        for word in shuffled:
            if word not in split:
                sizes = [0, 0]
                for known in classes.words[name]:
                    if known in split:
                        sizes[split[known]] += 1
                assign(word, 0 if sizes[0] <= sizes[1] else 1)
    return split


def is_bad(classes, target, triple, odd, odd_class, kind):
    listed = classes.member.get(odd, set())
    if target in listed or odd in triple:
        return True
    if any(odd_class in classes.member.get(word, set()) for word in triple):
        return True
    if kind == "unrelated":
        return any(classes.domain[m] == classes.domain[target] for m in listed)
    return False


def form_puzzles(classes, in_half, targets, odds_of, total, kind, rng):
    want, extra = divmod(total, len(targets))
    order = list(targets)
    rng.shuffle(order)
    puzzles = []
    seen = set()
    for i, target in enumerate(order):
        quota = want + (1 if i < extra else 0)
        odds = odds_of(target)
        made = 0
        for _ in range(quota * 500):
            if made == quota:
                break
            triple = rng.sample(in_half[target], 3)
            odd_class = rng.choice(odds)
            odd = rng.choice(in_half[odd_class])
            key = (frozenset(triple), odd)
            if key in seen or is_bad(classes, target, triple, odd, odd_class, kind):
                continue
            seen.add(key)
            words = [*triple, odd]
            rng.shuffle(words)
            puzzles.append((words, odd, target, odd_class, kind))
            made += 1
        if made < quota:
            raise ValueError(f"only {made} of {quota} {kind} puzzles for {target}")
    return puzzles


def half_puzzles(classes, split, half, rng):
    in_half = {}
    for name in sorted(classes.words):
        found = [word for word in classes.words[name] if split[word] == half]
        if found:
            in_half[name] = found
    nouns = [name for name in in_half if classes.part[name] == "noun"]
    puzzles = []
    for part in ("noun", "adjective", "verb"):
        pool = [
            name
            for name in in_half
            if classes.part[name] == part and classes.verb_side.get(name, half) == half
        ]
        full = [name for name in pool if len(in_half[name]) >= 3]

        def siblings(target, pool=pool):
            same = classes.domain[target]
            return [n for n in pool if n != target and classes.domain[n] == same]

        def strangers(target):
            same = classes.domain[target]
            return [n for n in nouns if classes.domain[n] != same]

        close = [name for name in full if siblings(name)]
        puzzles += form_puzzles(
            classes, in_half, close, siblings, SIZES[part], "close", rng
        )
        puzzles += form_puzzles(
            classes, in_half, full, strangers, SIZES[part], "unrelated", rng
        )
    return puzzles


def tagged_line(words, answer, target, odd_class, kind, classes, form, pair):
    tags = (
        f"part={classes.part[target]} type={kind} target={target.replace(' ', '_')} "
        f"odd={odd_class.replace(' ', '_')} domain={classes.domain[target]} "
        f"form={form} pair={pair}"
    )
    return f"{' '.join(words)} ; {answer} ; {tags}"


def twin_forms(classes, twins, words, answer, target, odd_class, kind):
    forms = []
    for mapping in (twins.us_form, twins.uk_form):
        mapped = [mapping.get(word, word) for word in words]
        odd = mapping.get(answer, answer)
        triple = [word for word in mapped if word != odd]
        if len(set(mapped)) < 4 or is_bad(
            classes, target, triple, odd, odd_class, kind
        ):
            return None
        forms.append((mapped, odd))
    return forms


def puzzle_lines(classes, twins, puzzles):
    lines = []
    counts = {"pairs": 0, "spelling": 0, "vocabulary": 0}
    for words, answer, target, odd_class, kind in puzzles:
        common = (target, odd_class, kind, classes)
        forms = None
        if any(word in twins.other for word in words):
            forms = twin_forms(classes, twins, words, answer, target, odd_class, kind)
        if forms is None:
            lines.append(tagged_line(words, answer, *common, "base", "-"))
            continue
        counts["pairs"] += 1
        pair = f"p{counts['pairs']:04d}"
        by_list = any(word in twins.vocabulary for word in words)
        counts["vocabulary" if by_list else "spelling"] += 1
        for form, (mapped, odd) in zip(("us", "uk"), forms):
            lines.append(tagged_line(mapped, odd, *common, form, pair))
    return lines, counts


def build_all(list_dir, out_dir, seed=780):
    classes = build_classes(list_dir, zipf_of)
    twins = load_twins(list_dir, zipf_of)
    split = split_words(classes, twins.other, seed)
    rng = random.Random(seed)
    counts = {
        "classes": {name: len(classes.words[name]) for name in sorted(classes.words)},
        "dropped": sorted(classes.dropped),
    }
    for half, (key, file) in enumerate(zip(("tune", "test"), HALF_FILES)):
        puzzles = half_puzzles(classes, split, half, rng)
        lines, found = puzzle_lines(classes, twins, puzzles)
        fileio.write_lines(os.path.join(out_dir, file), lines)
        words = {word for line in lines for word in line.split(" ; ")[0].split()}
        counts[key] = {
            "lines": len(lines),
            "puzzles": len(puzzles),
            "words": len(words),
            **found,
        }
    for name, size in counts["classes"].items():
        print(f"{name}: {size}")
    print("dropped under 6:", " ".join(counts["dropped"]) or "none")
    for key in ("tune", "test"):
        print(key, counts[key])
    return counts
