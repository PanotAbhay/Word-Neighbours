# Word-Neighbours

A distributional semantics model. It builds word vectors from word co-occurrence counts on large English text, with no neural network and no pre-built vectors, and it solves odd-one-out puzzles: give it `apple pear plum chair` and it answers `chair`.

## Use

You need Python 3.14 and a saved model folder holding `vectors.npy`, `words.txt`, and `meta.json`. Download the ready model (112 MB) from the [v0.1.0 release](https://github.com/PanotAbhay/Word-Neighbours/releases/tag/v0.1.0), unzip it, and move the `model` folder to `data/model`. You can also pass another path with `--model`.

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python cli.py
```

At the prompt:

- Type one word to see its 10 nearest neighbours.
- Type three or more words to get the odd one out.
- Type `q` to quit.

The prompt lists any word that is not in the vocabulary. The vocabulary is the 100,000 most frequent words of the text, chosen only by frequency, so it holds names and rude words from the web.

## Full build

This rebuilds the model and the puzzles from pinned sources. It needs about 34 GB of download, 16 GB of RAM, more disk for count tables, and up to 2 days of machine time.

```
.venv\Scripts\python -m pip install -r requirements-train.txt
.venv\Scripts\python cli.py download
.venv\Scripts\python cli.py puzzles
.venv\Scripts\python cli.py train
.venv\Scripts\python cli.py test
```

| Command | What it does |
|---|---|
| `download [--lists]` | Fetches the texts and word lists, resumes after a dropped connection, and checks each SHA-256 against `sources.json`. `--lists` fetches only the word lists. |
| `puzzles [--seed N]` | Builds `data/puzzles_tune.txt` and `data/puzzles_test.txt` from the word lists. The default seed is 780. Some seeds cannot fill the class quotas, and the command then asks for another seed. |
| `train` | Counts, tunes on the tune puzzles only, and saves the model. It exits with code 3 if the tune score stalls below 0.90. |
| `train --corpus DIR --out DIR` | Runs one count on a folder of `.txt` files, with no tuning. This is a smoke test. |
| `train --measure-only --words N` | Runs one count and one vector step on a prefix, and prints rates and peak RAM. |
| `test [--owner-approved]` | Scores a frozen model on the test puzzles and appends the result to `data/test_log.txt`. It refuses a model that is not frozen, a stalled model without `--owner-approved`, and a second model on the same test. |

## How it works

1. Counts words, and keeps the top 100,000 as rows.
2. Counts co-occurrence in a window with weight `K - d + 1` for distance `d`. Stopwords never become columns.
3. Converts counts to PPMI with context smoothing 0.75.
4. Reduces to about 300 numbers per word with randomized SVD, weighting by eigenvalues.
5. Normalises, mean-centres, and normalises again.

The odd one out is the word with the lowest total cosine to the others. A puzzle with an unknown word counts as wrong.

## What the test covers

A script builds every puzzle from published word lists. No person writes or edits a puzzle, and tuning never sees the test half. The test holds 2,000 base puzzles: 1,600 noun, 200 verb, and 200 adjective, each half close and half unrelated. Puzzles with a listed UK or US form appear twice, once in each form.

The score applies to this test. It is not a score on all English words. The test uses about 450 words of the 28,338 alpha words at Zipf 3 or above.

Known gaps:

1. **Modern words.** Only electronics, dish, and shop classes add them. `email`, `blog`, `podcast`, `selfie`, `emoji`, `wifi`, `internet`, `app`, and `browser` have no clean class. The `modern` probe tier in `data/probes.txt` is the only view of these words.
2. **UK/US vocabulary.** `voc.tab` has about 100 usable rows at most, so coverage is partial.
3. **Verbs.** The test holds only 45 verbs.
4. **Bias.** Battig is 1969 US data, and the Zipf 3 floor excludes some UK words such as `courgette` and `aubergine`.

Nouns are 80% of the test, and the pass rule is pooled, so read the verb and adjective scores next to the headline.

## Results

The target is 0.90 on the test puzzles. The model has not reached it, and the final test has not run.

Score on the tune puzzles (2,229 puzzles, 8.36 billion words, 100,000 words, 300 dimensions):

| Part | Score | Puzzles |
|---|---|---|
| Overall | 0.847 | 2,229 |
| Noun | 0.852 | 1,805 |
| Verb | 0.746 | 213 |
| Adjective | 0.900 | 211 |
| Unrelated odd word | 0.924 | 1,117 |
| Close odd word | 0.769 | 1,112 |

Learning curve on the tune puzzles: 0.825 at 100 million words, 0.841 at 1 billion words, 0.847 at 8.36 billion words.

A model without SVD scored 0.792, so the SVD step helps. A hubness correction (CSLS) lowered the score and the pipeline dropped it. One method round (shifted PPMI, `skip`, `p`, up to 600 dimensions, `rule`, 20,000 columns) gained at most 0.0009, below the 0.005 gain rule, so the model stayed the same.

Rerun on 2026-11-05: the saved model gave the same scores on the tune puzzles, and all 280 unit tests passed. No tune puzzle held a word outside the vocabulary. The US form scored 0.817 and the UK form scored 0.843 on 229 twin pairs.

The tune puzzles chose the settings, so this score is not an unbiased result. The test puzzles have not been scored. Close nouns cause most errors.

## Credits and notices

This is not legal advice. The saved model holds word vectors and not the text. Raw lists and generated puzzles are not in this repository, because their licences differ.

Text:

- [JoeyLLM/uk-dataset-1b](https://huggingface.co/datasets/JoeyLLM/uk-dataset-1b): CC BY 4.0 for the card and outputs. Original publishers may keep rights in the text. The card warns of possible personal information with no dedicated masking.
- [HuggingFaceFW/fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) `sample-10BT`: ODC-By. Common Crawl terms may also apply.

Word lists:

- [vecto-ai/word-benchmarks](https://github.com/vecto-ai/word-benchmarks): Battig, AP, BLESS, and ESSLLI 2008 lists. The repository has no licence field. Battig through WordPools is GPL-2.
- [wordfreq](https://github.com/rspeer/wordfreq) by Robyn Speer: used for the Zipf filter only. Its data is CC BY-SA 4.0. It draws on Google Books Ngram, SUBTLEX, and OpenSubtitles. Zenodo DOI 10.5281/zenodo.7199437.
- [Open English WordNet 2025](https://github.com/globalwordnet/english-wordnet): CC BY 4.0.
- [VerbNet 3](https://verbs.colorado.edu/verbnet_downloads/downloads.html) through NLTK data: Kipper, Korhonen, Ryant, and Palmer. The University of Colorado permission notice applies.
- [VarCon](https://github.com/en-wl/wordlist): the Kevin Atkinson and Titze notices apply.
