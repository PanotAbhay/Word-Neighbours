import hashlib
import http.server
import os
import random
import re
import tempfile
import threading
import unittest

from fetch import DownloadError, download_all, download_file

DATA = random.Random(780).randbytes(5 * 1024 * 1024)
SHA = hashlib.sha256(DATA).hexdigest()
MIB = 1024 * 1024


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        server = self.server
        server.requests += 1
        if server.always_503:
            self.send_error(503)
            return
        start, end = 0, len(DATA) - 1
        match = re.fullmatch(r"bytes=(\d+)-(\d*)", self.headers.get("Range", ""))
        ranged = match is not None and not server.ignore_range
        if ranged:
            start = int(match[1])
            end = min(int(match[2]) if match[2] else end, end)
            if server.first_range_start is None:
                server.first_range_start = start
        body = DATA[start : end + 1]
        self.send_response(206 if ranged else 200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if server.drop_after:
            self.wfile.write(body[: server.drop_after])
            self.wfile.flush()
            self.close_connection = True
        else:
            self.wfile.write(body)


class FetchTest(unittest.TestCase):
    def setUp(self):
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.requests = 0
        self.server.always_503 = False
        self.server.ignore_range = False
        self.server.drop_after = 0
        self.server.first_range_start = None
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = folder.name
        self.dest = os.path.join(self.folder, "file.bin")
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/file.bin"

    def fetch(self, sha=SHA, **kwargs):
        kwargs.setdefault("chunk_bytes", MIB)
        kwargs.setdefault("wait", 0)
        download_file(self.url, self.dest, len(DATA), sha, **kwargs)

    def stored(self):
        with open(self.dest, "rb") as f:
            return f.read()

    def test_resume_after_drops(self):
        self.server.drop_after = 300 * 1024
        self.fetch()
        self.assertEqual(self.stored(), DATA)
        self.assertGreater(self.server.requests, 5)

    def test_server_ignores_range(self):
        self.server.ignore_range = True
        self.fetch()
        self.assertEqual(self.stored(), DATA)

    def test_server_ignores_range_with_part_file(self):
        self.server.ignore_range = True
        with open(self.dest + ".part", "wb") as f:
            f.write(DATA[: 2 * MIB])
        self.fetch()
        self.assertEqual(self.stored(), DATA)

    def test_wrong_hash_fails_and_removes_part(self):
        with self.assertRaises(DownloadError):
            self.fetch(sha="0" * 64)
        self.assertFalse(os.path.exists(self.dest))
        self.assertFalse(os.path.exists(self.dest + ".part"))

    def test_resume_across_runs(self):
        with open(self.dest + ".part", "wb") as f:
            f.write(DATA[: 2 * MIB])
        self.fetch()
        self.assertEqual(self.server.first_range_start, 2 * MIB)
        self.assertEqual(self.stored(), DATA)

    def test_part_longer_than_size_restarts(self):
        with open(self.dest + ".part", "wb") as f:
            f.write(DATA + b"extra")
        self.fetch()
        self.assertEqual(self.stored(), DATA)

    def test_complete_file_not_refetched(self):
        with open(self.dest, "wb") as f:
            f.write(DATA)
        self.fetch()
        self.assertEqual(self.server.requests, 0)

    def test_gives_up_after_tries(self):
        self.server.always_503 = True
        with self.assertRaises(DownloadError):
            self.fetch(tries=2)
        self.assertEqual(self.server.requests, 2)

    def test_unknown_size_checks_hash_only(self):
        download_file(self.url, self.dest, None, SHA, chunk_bytes=MIB, wait=0)
        self.assertEqual(self.stored(), DATA)

    def test_download_all_places_files_and_records_time(self):
        sources = {
            "lists": [
                {
                    "name": "a.bin",
                    "url": self.url,
                    "size": len(DATA),
                    "sha256": SHA,
                    "group": None,
                }
            ],
            "text": [
                {
                    "name": "b.bin",
                    "url": self.url,
                    "size": len(DATA),
                    "sha256": SHA,
                    "group": "uk",
                }
            ],
        }
        root = os.path.join(self.folder, "raw")
        download_all(sources, root, kinds=("lists",))
        self.assertTrue(os.path.exists(os.path.join(root, "lists", "a.bin")))
        self.assertFalse(os.path.exists(os.path.join(root, "text", "uk", "b.bin")))
        download_all(sources, root)
        self.assertTrue(os.path.exists(os.path.join(root, "text", "uk", "b.bin")))
        self.assertTrue(
            os.path.exists(os.path.join(self.folder, "work", "time_used.json"))
        )


if __name__ == "__main__":
    unittest.main()
