import hashlib
import json
import os
import time
from http.client import HTTPException
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

HEADERS = {"User-Agent": "word-neighbours-download/1.0"}
RETRY_CODES = (429, 500, 502, 503, 504)


class DownloadError(Exception):
    pass


def load_sources(path="sources.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def fetch_range(url, part, start, end):
    headers = dict(HEADERS)
    if end is None:
        headers["Range"] = f"bytes={start}-"
    else:
        headers["Range"] = f"bytes={start}-{end}"
    with urlopen(Request(url, headers=headers), timeout=180) as response:
        if response.status == 206:
            mode, position = "ab", start
        else:
            mode, position = "wb", 0
        total = response.headers.get("Content-Length")
        with open(part, mode) as f:
            got = 0
            while block := response.read(1024 * 1024):
                f.write(block)
                got += len(block)
        if total is not None and got < int(total):
            raise OSError(f"connection closed after {got} of {total} bytes")
        return position + got, response.status


def download_file(
    url, dest, size, sha256, chunk_bytes=32 * 1024 * 1024, tries=12, wait=2.0
):
    if os.path.exists(dest) and (size is None or os.path.getsize(dest) == size):
        return
    part = dest + ".part"
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    if os.path.exists(part) and size is not None and os.path.getsize(part) > size:
        os.remove(part)
    failures = 0
    while True:
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if size is not None and have == size:
            break
        end = have + chunk_bytes - 1
        if size is not None:
            end = min(end, size - 1)
        try:
            position, status = fetch_range(url, part, have, end)
        except HTTPError as e:
            e.close()
            if e.code == 416 and size is None and have > 0:
                break
            if e.code not in RETRY_CODES:
                raise DownloadError(f"{url}: HTTP {e.code}") from e
            position, status = None, None
        except (HTTPException, URLError, OSError):
            position, status = None, None
        now = os.path.getsize(part) if os.path.exists(part) else 0
        if position is None and now <= have:
            failures += 1
            if failures >= tries:
                raise DownloadError(f"{url}: gave up after {tries} tries")
            time.sleep(wait)
            continue
        failures = 0
        if size is None and status == 200 and position is not None:
            break
        if size is None and position is not None and now - have < chunk_bytes:
            break
    if size is not None and os.path.getsize(part) != size:
        os.remove(part)
        raise DownloadError(f"{url}: wrong size")
    if sha256_of(part) != sha256:
        os.remove(part)
        raise DownloadError(f"{url}: wrong SHA-256")
    os.replace(part, dest)


def add_time_used(root, key, seconds):
    path = os.path.join(root, "..", "work", "time_used.json")
    path = os.path.normpath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    used = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            used = json.load(f)
    used[key] = used.get(key, 0.0) + seconds
    with open(path, "w", encoding="utf-8", newline="") as f:
        json.dump(used, f)


def download_all(sources, root, kinds=("lists", "text")):
    started = time.monotonic()
    for kind in kinds:
        for entry in sources[kind]:
            if kind == "lists":
                dest = os.path.join(root, "lists", entry["name"])
            else:
                dest = os.path.join(root, "text", entry["group"], entry["name"])
            download_file(entry["url"], dest, entry["size"], entry["sha256"])
            print(f"ok {dest}", flush=True)
    add_time_used(root, "download", time.monotonic() - started)
