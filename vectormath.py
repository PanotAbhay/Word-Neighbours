import math


def dot(u, v):
    dot_product = 0.0
    for u_i, v_i in zip(u, v):
        dot_product += u_i * v_i
    return float(dot_product)


def norm(v):
    return math.sqrt(dot(v, v))


def cosine(u, v):
    norm_u, norm_v = norm(u), norm(v)
    if norm_u == 0.0 or norm_v == 0.0:
        return 0.0
    similarity = dot(u, v) / (norm_u * norm_v)
    return max(-1.0, min(1.0, similarity))


def accuracy(predicted, actual):
    correct = 0
    for guess, answer in zip(predicted, actual):
        if guess is not None and guess == answer:
            correct += 1
    return correct / len(actual) if actual else 0.0


if __name__ == "__main__":
    a = [2, 0, 1]
    b = [1, 1, 1]
    c = [0, 0, 0]

    print(f"{dot(a, b) = }")
    print(f"{norm(a) = }")
    print(f"{cosine(a, b) = }")
    print(f"{cosine(a, a) = }")
    print(f"{cosine(a, c) = }")
    print(f"{accuracy(['cat', 'dog', None], ['cat', 'cow', 'dog']) = }")
