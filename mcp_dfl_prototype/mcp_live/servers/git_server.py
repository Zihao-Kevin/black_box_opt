import subprocess
from mcp.server.mcpserver import MCPServer
from _common import WS, clip

server = MCPServer("git", instructions="Reads the git history of the workspace repository (log, show).")


def _git(*args):
    r = subprocess.run(["git", *args], cwd=WS, capture_output=True, text=True, timeout=20)
    return clip((r.stdout + r.stderr).strip() or "(no output)")


@server.tool()
def git_log(path: str = "", n: int = 15) -> str:
    """Commit history (hash, author, date, subject), newest first; optionally only commits touching `path`."""
    args = ["log", f"-n{n}", "--format=%h  %an <%ae>  %ad  %s", "--date=short"]
    if path:
        args += ["--", path]
    return _git(*args)


@server.tool()
def git_show(rev: str = "HEAD") -> str:
    """Show one commit: metadata and the list of files it changed."""
    if rev.startswith("-") or ":" in rev:
        raise ValueError("rev must be a commit reference")
    return _git("show", "--stat", "--format=%H%nAuthor: %an <%ae>%nDate: %ad%n%n%s%n%n%b", "--date=short", rev)


if __name__ == "__main__":
    server.run(transport="stdio")
