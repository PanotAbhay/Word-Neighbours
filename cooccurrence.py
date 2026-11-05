def context_pairs(sequence, window):
    pairs = []
    for i in range(len(sequence)):
        for j in range(max(0, i - window), min(len(sequence), i + window + 1)):
            if i != j:
                pairs.append((sequence[i], sequence[j]))
    return pairs


def cooccurrence_counts(pairs):
    counts = {}
    for pair in pairs:
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def most_common(counts, n):
    items = list(counts.items())
    items.sort(key=lambda item: item[0])
    items.sort(key=lambda item: item[1], reverse=True)
    return items[:n]


def first_index(sequence, item):
    for i in range(len(sequence)):
        if sequence[i] == item:
            return i
    return -1


if __name__ == "__main__":
    from textbasics import tokenize

    with open("alice.txt", encoding="utf-8") as f:
        tokens = tokenize(f.read())

    pairs = context_pairs(tokens, 2)
    counts = cooccurrence_counts(pairs)

    print(f"{len(tokens) = }")
    print(f"{len(pairs) = }")
    print(f"{pairs[:6] = }")
    print(f"{most_common(counts, 5) = }")
    print(f"{first_index(tokens, 'alice') = }")
