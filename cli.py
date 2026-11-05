import argparse
import os
import string
import sys

import fileio
from model import ModelError, load_model, most_similar, odd_one_out

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RAW = os.path.join(DATA, "raw")


def load_puzzles(path):
    puzzles = []
    for line in fileio.read_lines(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split(";")
        words = fields[0].casefold().split()
        answer = fields[1].strip().casefold()
        puzzles.append((words, answer))
    return puzzles


def clean_input(line):
    words = []
    for piece in line.casefold().replace(",", " ").split():
        word = piece.strip(string.punctuation)
        if word:
            words.append(word)
    return words


def explore(model):
    print(f"{len(model.words)} words. Type a word, or several words, or q.")
    while True:
        try:
            line = input("> ").strip().casefold()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line == "q" or line == "":
            break
        words = clean_input(line)
        unknown = [word for word in words if word not in model.index]
        if len(words) == 0:
            print("  type a word, or several words, or q")
        elif len(unknown) > 0:
            print("  not in the vocabulary:", " ".join(unknown))
        elif len(words) == 1:
            for other, score in most_similar(words[0], model, 10):
                print(f"  {other:15} {score:.3f}")
        elif len(words) == 2:
            print("  a puzzle needs at least three words")
        elif len(set(words)) < len(words):
            print("  a word repeats, so type each word once")
        else:
            print("  odd one out:", odd_one_out(words, model))


def corpus_files(folder):
    paths = []
    for name in sorted(os.listdir(folder)):
        if name.endswith(".txt"):
            paths.append(os.path.join(folder, name))
    return paths


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Odd-one-out word model.")
    parser.add_argument("--model", default=os.path.join(DATA, "model"))
    commands = parser.add_subparsers(dest="command")
    download = commands.add_parser("download")
    download.add_argument("--lists", action="store_true")
    puzzles = commands.add_parser("puzzles")
    puzzles.add_argument("--seed", type=int, default=780)
    train = commands.add_parser("train")
    train.add_argument("--corpus")
    train.add_argument("--out")
    train.add_argument("--measure-only", action="store_true")
    train.add_argument("--words", type=int, default=100000000)
    train.add_argument("--method-round", action="store_true")
    test = commands.add_parser("test")
    test.add_argument("--owner-approved", action="store_true")
    return parser.parse_args(argv)


def run_download(lists_only):
    from fetch import DownloadError, download_all, load_sources

    kinds = ("lists",) if lists_only else ("lists", "text")
    try:
        download_all(load_sources(os.path.join(HERE, "sources.json")), RAW, kinds=kinds)
    except DownloadError as e:
        print(f"Download failed: {e}")
        return 1
    return 0


def run_puzzles(seed):
    from puzzlebuild import build_all

    try:
        build_all(os.path.join(RAW, "lists"), DATA, seed=seed)
    except ValueError as e:
        print(f"seed {seed} cannot fill the puzzle sizes, so try another seed: {e}")
        return 1
    return 0


def run_test_command(model_folder, owner_approved):
    from scoring import TestRefused, format_report, run_test

    puzzles = os.path.join(DATA, "puzzles_test.txt")
    log = os.path.join(DATA, "test_log.txt")
    try:
        report = run_test(model_folder, puzzles, log, owner_approved=owner_approved)
    except (TestRefused, ModelError, OSError) as e:
        print(f"The test did not run: {e}")
        return 1
    print(format_report(report))
    return 0


def run_train(args):
    from concurrent.futures.process import BrokenProcessPool

    from counting import CountSettings
    from tune import (
        TrainError,
        measure_only,
        run_method_round,
        train_corpus,
        train_full,
    )
    from vectors import VectorSettings

    out = args.out or (os.path.join(DATA, "smoke") if args.corpus else args.model)
    work = os.path.join(DATA, "work")
    try:
        if args.corpus:
            paths = corpus_files(args.corpus)
            train_corpus(paths, CountSettings(), VectorSettings(), out)
            return 0
        if args.measure_only:
            measure_only(args.words, work)
            return 0
        if args.method_round:
            result = run_method_round(work, out)
        else:
            result = train_full(work, out)
    except TrainError as e:
        print(f"Training did not run: {e}")
        return 1
    except BrokenProcessPool:
        print("A worker process ended. Run the command again to resume.")
        return 1
    return 0 if result["decision"] == "freeze" else 3


def main(argv):
    args = parse_args(argv)
    if args.command == "download":
        return run_download(args.lists)
    if args.command == "puzzles":
        return run_puzzles(args.seed)
    if args.command == "test":
        return run_test_command(args.model, args.owner_approved)
    if args.command == "train":
        return run_train(args)
    try:
        model = load_model(args.model)
    except ModelError as e:
        print(f"No usable model: {e}")
        print("Download a model, or run:  python cli.py train")
        return 1
    explore(model)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
