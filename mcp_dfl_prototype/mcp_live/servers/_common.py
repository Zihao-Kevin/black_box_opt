"""Shared helpers for the stdio MCP servers.  Every server is started with WORKSPACE=<dir> in its environment and
may only touch files inside that directory; dot-files (.git, .outbox.json, .tickets.json, .notes.json) are the
harness's own state and are hidden from the file-oriented servers."""
import os, json, threading

WS = os.environ.get("WORKSPACE", os.getcwd())


def safe(rel):
    """Resolve a workspace-relative path; refuse traversal, absolute paths and dot-files."""
    rel = (rel or "").strip().lstrip("./")
    if not rel or rel.startswith("/") or ".." in rel.split("/") or any(p.startswith(".") for p in rel.split("/")):
        raise ValueError(f"path not allowed: {rel!r}")
    return os.path.join(WS, rel)


def visible_files():
    out = []
    for root, dirs, files in os.walk(WS):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d != "__pycache__")
        for f in sorted(files):
            if not f.startswith(".") and not f.endswith(".pyc"):
                out.append(os.path.relpath(os.path.join(root, f), WS))
    return out


_lock = threading.Lock()


def append_json(dotfile, record):
    """Append a record to a hidden JSON list in the workspace (mail outbox, ticket tracker)."""
    path = os.path.join(WS, dotfile)
    with _lock:
        data = json.load(open(path)) if os.path.exists(path) else []
        data.append(record)
        json.dump(data, open(path, "w"), indent=1)
    return len(data)


def clip(text, n=4000):
    return text if len(text) <= n else text[:n] + f"\n... [{len(text) - n} more characters truncated]"
