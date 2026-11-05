def read_text(path):
    with open(path, encoding="utf-8-sig") as f:
        return f.read()


def read_lines(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        lines = f.read().split("\n")
    if lines[-1] == "":
        lines.pop()
    return [line.removesuffix("\r") for line in lines]


def write_lines(path, lines):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.writelines(line + "\n" for line in lines)


def append_line(path, line):
    with open(path, "a", encoding="utf-8", newline="") as f:
        f.write(line + "\n")


if __name__ == "__main__":
    text = read_text("alice.txt")
    lines = read_lines("alice.txt")
    print(f"{len(text) = }")
    print(f"{len(lines) = }")
    print(f"{lines[:2] = }")

    write_lines("sample.txt", ["first", "second"])
    append_line("sample.txt", "third")
    print(f"{read_lines('sample.txt') = }")

    append_line("log.txt", "ran fileio.py")
    print(f"{len(read_lines('log.txt')) = } run(s) so far")
