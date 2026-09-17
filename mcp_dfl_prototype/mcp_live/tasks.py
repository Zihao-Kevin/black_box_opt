"""Ten tasks for the agent, each with a programmatic evaluation oracle.  `need` is the designer's expectation of the
servers required (used only for diagnostics: the reward comes from the oracle, never from `need`)."""
import os, json, re, hashlib


def _answer(final_text):
    """The value on the agent's last `ANSWER:` line, or the whole final message if it gave none."""
    m = re.findall(r"ANSWER\s*:\s*(.+)", final_text or "", flags=re.I)
    return (m[-1] if m else (final_text or "")).strip().strip("`*'\" .")


def _num(s):
    return re.sub(r"[,\s]", "", s)


def _outbox(ws):
    p = os.path.join(ws, ".outbox.json"); return json.load(open(p)) if os.path.exists(p) else []


def _tickets(ws):
    p = os.path.join(ws, ".tickets.json"); return json.load(open(p)) if os.path.exists(p) else []


def _file(ws, rel):
    p = os.path.join(ws, rel); return open(p).read() if os.path.exists(p) else None


def make_tasks(facts, intranet):
    """facts: from workspace.build_template(); intranet: base URL of the docs site."""
    digest12 = hashlib.sha256(facts["secret"].encode()).hexdigest()[:12]
    tasks = [
        dict(id="port", need=["filesystem"],
             spec="What TCP port does the service listen on according to config.ini in the workspace? Reply with the port number.",
             oracle=lambda ws, out: _num(_answer(out)) == facts["port"]),
        dict(id="shipped", need=["sqlite", "write_file"],
             spec="How many orders in the project database data.db have status 'shipped'? Write just that number into a new file answer.txt in the workspace.",
             oracle=lambda ws, out: _num((_file(ws, "answer.txt") or "").strip()) == facts["shipped"]),
        dict(id="retries", need=["fetch"],
             spec=f"The vendor SDK docs page {intranet}/docs/retries.html states how many times a failed request is retried by default. Reply with that number.",
             oracle=lambda ws, out: _num(_answer(out)) == facts["retries"]),
        dict(id="digest", need=["filesystem", "python"],
             spec="Compute the SHA-256 hex digest of the exact contents of secret.txt in the workspace (the file has no trailing newline). Reply with the first 12 hex characters of the digest.",
             oracle=lambda ws, out: _answer(out).lower().startswith(digest12)),
        dict(id="passing", need=["pytest"],
             spec="Run the project's test suite and reply with the number of tests that pass.",
             oracle=lambda ws, out: _num(_answer(out)) == facts["n_pass"]),
        dict(id="author", need=["git", "mail"],
             spec=f"Find the author of the most recent git commit that changed {facts['defines_file']} and e-mail that person's full name to eng-leads@corp.example with the subject 'ratio owner'.",
             oracle=lambda ws, out: any(m["to"].strip().lower() == "eng-leads@corp.example" and facts["ratio_author"].lower() in (m["subject"] + " " + m["body"]).lower() for m in _outbox(ws))),
        dict(id="ratelimit", need=["fetch", "ticket"],
             spec=f"The intranet page {intranet}/docs/limits.html states the vendor's per-minute request limit. Open a ticket whose title contains that number of requests per minute.",
             oracle=lambda ws, out: any(facts["ratelimit"] in _num(t["title"]) for t in _tickets(ws))),
        dict(id="defines", need=["grep"],
             spec="Which file in the workspace defines the function normalize_ratio? Reply with its path relative to the workspace root.",
             oracle=lambda ws, out: _answer(out).replace("\\", "/").strip("/") .endswith(facts["defines_file"])),
        dict(id="lint", need=["lint"],
             spec="How many warnings does the static linter (pyflakes) report for src/mod_09.py? Reply with the number.",
             oracle=lambda ws, out: _num(_answer(out)) == facts["lint_warnings"]),
        dict(id="oncall", need=["notes", "mail"],
             spec="According to the team notes, who is the primary on-call engineer for the week of 2026-09-07? E-mail that engineer's full name to oncall@corp.example with the subject 'on-call check'.",
             oracle=lambda ws, out: any(m["to"].strip().lower() == "oncall@corp.example" and facts["oncall_primary"].lower() in (m["subject"] + " " + m["body"]).lower() for m in _outbox(ws))),
    ]
    return tasks
