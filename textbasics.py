PUNCTUATION = ".,;:!?'\"()-—_“”‘’"
PUNCTUATION_TO_SPACE = str.maketrans(PUNCTUATION, " " * len(PUNCTUATION))


def tokenize(text):
    text = text.casefold().translate(PUNCTUATION_TO_SPACE)
    tokens = text.split()
    return tokens


def count_words(tokens):
    counts = {}
    for word in tokens:
        counts[word] = counts.get(word, 0) + 1
    return counts


def build_vocabulary(tokens):
    vocabulary = {}
    for word in tokens:
        if word not in vocabulary:
            vocabulary[word] = len(vocabulary)
    return vocabulary


def encode(tokens, vocabulary):
    ids = []
    for word in tokens:
        ids.append(vocabulary[word])
    return ids


def decode(ids, vocabulary):
    id_to_word = {}
    for word, value in vocabulary.items():
        id_to_word[value] = word
    words = []
    for token_id in ids:
        words.append(id_to_word[token_id])
    return words


if __name__ == "__main__":
    text = """The cat sat on the mat.
The cat sat again!"""

    tokens = tokenize(text)
    counts = count_words(tokens)
    vocabulary = build_vocabulary(tokens)
    ids = encode(tokens, vocabulary)

    print(f"{tokens = }")
    print(f"{counts = }")
    print(f"{vocabulary = }")
    print(f"{ids = }")
    print(f"{decode(ids, vocabulary) = }")
