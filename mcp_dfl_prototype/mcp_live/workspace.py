"""The project workspace every agent episode runs in, built deterministically once (`build_template`) and copied
fresh for every episode (`fresh_copy`), plus the intranet the `fetch` server talks to (`start_intranet`)."""
import os, json, random, shutil, sqlite3, subprocess, tempfile, threading, http.server, socketserver, functools, sys

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "_workspace_template")
WWW = os.path.join(HERE, "_www")

AUTHORS = [("Mara Okafor", "mara@corp.example"), ("Deniz Kaya", "deniz@corp.example"), ("Priya Natarajan", "priya@corp.example")]

UTILS = '''"""Small helpers used across the service."""
import re

_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def parse_duration(text):
    """'90s' -> 90, '2m' -> 120, '1h30m' -> 5400, '1d' -> 86400."""
    total = 0
    for num, unit in re.findall(r"(\\d+)([smhd])", text):
        if unit == "h":            # BUG: hours are counted as minutes
            total += int(num) * 60
        else:
            total += int(num) * _UNITS[unit]
    return total


def slugify(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
'''

TEST = '''import json, os
import pytest
from utils import parse_duration

CASES = json.load(open(os.path.join(os.path.dirname(__file__), "cases.json")))


@pytest.mark.parametrize("text,expected", CASES, ids=[c[0] for c in CASES])
def test_parse_duration(text, expected):
    assert parse_duration(text) == expected
'''

CONFIG = """[server]
host = 0.0.0.0
port = 8443
workers = 4

[retry]
attempts = 3
backoff = 2.0

[db]
path = data.db
"""

NOTES = [
    {"title": "On-call rotation", "body": "Week of 2026-08-31: primary Mara Okafor, secondary Deniz Kaya.\n"
                                          "Week of 2026-09-07: primary Priya Natarajan, secondary Mara Okafor.\n"
                                          "Week of 2026-09-14: primary Deniz Kaya, secondary Priya Natarajan."},
    {"title": "Deploy checklist", "body": "1. run the test suite  2. bump the version  3. tag the release  4. announce in #eng."},
    {"title": "Retry policy decision (2026-06)", "body": "We keep the vendor default retry count rather than overriding it in config.ini."},
    {"title": "Database conventions", "body": "Amounts are stored in cents. Order status is one of: pending, shipped, delivered, cancelled."},
]

DOCS = {
    "index.html": "<h1>Vendor SDK documentation</h1><ul><li><a href='retries.html'>Retries</a></li><li><a href='limits.html'>Rate limits</a></li><li><a href='auth.html'>Authentication</a></li></ul>",
    "retries.html": "<h1>Retries</h1><p>The client transparently retries failed requests with exponential backoff.</p>"
                    "<p>By default a request is retried <b>5</b> times before the error is raised; the maximum configurable value is 10 and the initial backoff is 250 ms.</p>",
    "limits.html": "<h1>Rate limits</h1><p>Each API key may issue at most <b>120 requests per minute</b> (a burst of 40 is allowed). "
                   "Daily quota: 50,000 requests. Exceeding the limit returns HTTP 429.</p>",
    "auth.html": "<h1>Authentication</h1><p>Pass the API key in the <code>X-Api-Key</code> header. Keys expire after 90 days.</p>",
}

FUNCS = ["clamp", "chunk", "flatten", "pairwise", "dedupe", "rolling_mean", "parse_bool", "to_camel", "to_snake", "safe_div",
         "percentile", "histogram", "argmax", "topk", "moving_max", "zscore", "normalize_ratio", "levenshtein", "wrap_text",
         "hex_dump", "merge_dicts", "deep_get", "batched", "retry_call"]


def _module(i, fname):
    return (f'"""Module {i:02d}."""\n\n\ndef {fname}(x, y=None):\n    """Helper {fname}."""\n    if y is None:\n        return x\n'
            f'    return (x, y)\n\n\ndef _helper_{i:02d}(v):\n    return v * {i}\n')


LEGACY = '''"""Legacy helpers (unmaintained)."""
import os
import sys
import json
from collections import OrderedDict


def load(path):
    data = json.loads(open(path).read())
    unused = 1
    return data


def total(items):
    result = 0
    for item in items:
        result += item.value
    return reslt


def dump(obj):
    return json.dumps(obj, indent=2)
'''


def _git(ws, *args, author=None):
    env = dict(os.environ)
    if author:
        env.update(GIT_AUTHOR_NAME=author[0], GIT_AUTHOR_EMAIL=author[1], GIT_COMMITTER_NAME=author[0], GIT_COMMITTER_EMAIL=author[1])
    subprocess.run(["git", *args], cwd=ws, env=env, check=True, capture_output=True)


def build_template(force=False):
    """Create the workspace template (idempotent). Returns a dict of the facts the oracles check."""
    facts_path = os.path.join(TEMPLATE, ".facts.json")
    if os.path.exists(facts_path) and not force:
        return json.load(open(facts_path))
    shutil.rmtree(TEMPLATE, ignore_errors=True); shutil.rmtree(WWW, ignore_errors=True)
    os.makedirs(os.path.join(TEMPLATE, "src")); os.makedirs(os.path.join(TEMPLATE, "tests")); os.makedirs(os.path.join(WWW, "docs"))
    rng = random.Random(7)
    w = lambda rel, text: open(os.path.join(TEMPLATE, rel), "w").write(text)
    w("README.md", "# orders-service\n\nInternal order-tracking service. See config.ini for runtime settings, "
                   "src/ for helpers, tests/ for the test suite. Data lives in data.db.\n")
    w("config.ini", CONFIG)
    secret = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(40))
    w("secret.txt", secret)                                                    # no trailing newline
    w("utils.py", UTILS); w("conftest.py", "")
    w("tests/test_parse_duration.py", TEST)
    cases = []
    for i in range(40):
        parts = rng.sample(["s", "m", "h", "d"], rng.randint(1, 2))
        text = "".join(f"{rng.randint(1, 9)}{u}" for u in parts)
        secs = sum(int(n) * {"s": 1, "m": 60, "h": 3600, "d": 86400}[u] for n, u in __import__("re").findall(r"(\d+)([smhd])", text))
        cases.append([text, secs])
    json.dump(cases, open(os.path.join(TEMPLATE, "tests", "cases.json"), "w"))
    n_pass = sum(1 for t, _ in cases if "h" not in t)
    for i, fname in enumerate(FUNCS, 1):
        w(f"src/mod_{i:02d}.py", _module(i, fname))
    w("src/mod_09.py", LEGACY)                                                  # the file with lint warnings
    defines_file = f"src/mod_{FUNCS.index('normalize_ratio') + 1:02d}.py"
    json.dump(NOTES, open(os.path.join(TEMPLATE, ".notes.json"), "w"), indent=1)
    # database
    db = sqlite3.connect(os.path.join(TEMPLATE, "data.db"))
    db.execute("CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, region TEXT)")
    db.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, status TEXT, amount_cents INTEGER, created TEXT)")
    names = ["Acme Corp", "Globex", "Initech", "Umbrella", "Hooli", "Vandelay", "Stark Industries", "Wonka", "Tyrell", "Cyberdyne"]
    for i, nm in enumerate(names, 1):
        db.execute("INSERT INTO customers VALUES (?,?,?)", (i, nm, rng.choice(["EMEA", "AMER", "APAC"])))
    statuses = ["pending", "shipped", "delivered", "cancelled"]
    for i in range(1, 241):
        db.execute("INSERT INTO orders VALUES (?,?,?,?,?)", (i, rng.randint(1, 10), rng.choices(statuses, [3, 4, 2, 1])[0],
                                                             rng.randint(500, 90000), f"2026-{rng.randint(1, 8):02d}-{rng.randint(1, 28):02d}"))
    db.commit()
    shipped = db.execute("SELECT COUNT(*) FROM orders WHERE status='shipped'").fetchone()[0]
    top = db.execute("SELECT c.name FROM orders o JOIN customers c ON c.id=o.customer_id GROUP BY c.id ORDER BY COUNT(*) DESC, c.name LIMIT 1").fetchone()[0]
    db.close()
    # intranet docs
    for name, html in DOCS.items():
        open(os.path.join(WWW, "docs", name), "w").write(f"<html><body>{html}</body></html>")
    # git history: several authors; the last commit touching the ratio module is by AUTHORS[1]
    _git(TEMPLATE, "init", "-q"); _git(TEMPLATE, "config", "commit.gpgsign", "false")
    _git(TEMPLATE, "add", "-A"); _git(TEMPLATE, "commit", "-q", "-m", "initial import", author=AUTHORS[0])
    w(defines_file, _module(FUNCS.index("normalize_ratio") + 1, "normalize_ratio").replace("Helper", "Normalise a ratio into [0, 1]. Helper"))
    _git(TEMPLATE, "commit", "-q", "-am", "ratio: document normalize_ratio", author=AUTHORS[2])
    w("README.md", open(os.path.join(TEMPLATE, "README.md")).read() + "\nRun `pytest` before opening a pull request.\n")
    _git(TEMPLATE, "commit", "-q", "-am", "README: mention the test suite", author=AUTHORS[0])
    w(defines_file, open(os.path.join(TEMPLATE, defines_file)).read().replace("return (x, y)", "return (x, y)  # clamp later"))
    _git(TEMPLATE, "commit", "-q", "-am", "ratio: note the clamping follow-up", author=AUTHORS[1])
    w("utils.py", UTILS.replace("Small helpers", "Small helpers (v2)"))
    _git(TEMPLATE, "commit", "-q", "-am", "utils: docstring", author=AUTHORS[2])
    facts = dict(port="8443", shipped=str(shipped), top_customer=top, retries="5", ratelimit="120", secret=secret,
                 n_pass=str(n_pass), n_tests=str(len(cases)), ratio_author=AUTHORS[1][0], defines_file=defines_file,
                 oncall_primary="Priya Natarajan")
    r = subprocess.run([sys.executable, "-m", "pyflakes", os.path.join(TEMPLATE, "src/mod_09.py")], capture_output=True, text=True)
    facts["lint_warnings"] = str(len([l for l in r.stdout.splitlines() if l.strip()]))
    json.dump(facts, open(facts_path, "w"), indent=1)
    return facts


def fresh_copy():
    d = tempfile.mkdtemp(prefix="mcp_ws_")
    shutil.rmtree(d); shutil.copytree(TEMPLATE, d, symlinks=True)
    os.remove(os.path.join(d, ".facts.json"))
    return d


_intranet = {}


def start_intranet(port=8765):
    """Serve `_www` on localhost in a daemon thread; returns the base URL (reused if already running)."""
    if "url" in _intranet:
        return _intranet["url"]
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a, **k):
            pass
    handler = functools.partial(Quiet, directory=WWW)
    while True:
        try:
            srv = socketserver.ThreadingTCPServer(("127.0.0.1", port), handler); break
        except OSError:                                   # port taken: reuse it if another process already serves our docs there
            try:
                import urllib.request
                if b"Vendor SDK" in urllib.request.urlopen(f"http://127.0.0.1:{port}/docs/index.html", timeout=2).read():
                    _intranet["url"] = f"http://127.0.0.1:{port}"; return _intranet["url"]
            except Exception:
                pass
            port += 1
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    _intranet.update(url=f"http://127.0.0.1:{port}", server=srv)
    return _intranet["url"]
