import textbasics
import cooccurrence


def split_sentences(text):
    text = text.replace("!", ".").replace("?", ".")
    sentences = []
    for piece in text.split("."):
        sentence = piece.strip()
        if sentence:
            sentences.append(sentence)
    return sentences


def clean(sentence):
    return "".join(ch if ch.isalpha() else " " for ch in sentence)


def tokenize_sentences(sentences):
    result = []
    for sentence in sentences:
        tokens = textbasics.tokenize(sentence)
        if tokens:
            result.append(tokens)
    return result


def flatten(nested):
    flat = []
    for inner in nested:
        flat.extend(inner)
    return flat


def remove_words(tokens, banned):
    return [token for token in tokens if token not in banned]


def select_context_words(counts, n):
    return [word for word, count in cooccurrence.most_common(counts, n)]


def count_cooccurrences(sentences, window):
    count = {}
    for sentence in sentences:
        for pair in cooccurrence.context_pairs(sentence, window):
            count[pair] = count.get(pair, 0) + 1
    return count


def old_counts(sentences, stopwords, window, n_context):
    cleaned = [remove_words(sentence, stopwords) for sentence in sentences]
    counts = textbasics.count_words(flatten(cleaned))
    words = list(counts)
    words_set = set(words)
    for sentence in sentences:
        for token in sentence:
            if token in stopwords and token not in words_set:
                words.append(token)
                words_set.add(token)
    context_words = select_context_words(counts, n_context)
    cooc = {}
    for w in range(1, window + 1):
        for pair, count in count_cooccurrences(cleaned, w).items():
            cooc[pair] = cooc.get(pair, 0) + count
    for w in range(1, window + 1):
        for pair, count in count_cooccurrences(sentences, w).items():
            if pair[0] in stopwords:
                cooc[pair] = cooc.get(pair, 0) + count
    return cooc, context_words, words
