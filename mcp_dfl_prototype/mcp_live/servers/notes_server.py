import os, json
from mcp.server.mcpserver import MCPServer
from _common import WS, clip

server = MCPServer("notes", instructions="Searches the team's shared notes (rotations, runbooks, decisions).")
NOTES = os.path.join(WS, ".notes.json")


def _notes():
    return json.load(open(NOTES)) if os.path.exists(NOTES) else []


@server.tool()
def list_notes() -> str:
    """List the titles of all team notes."""
    return "\n".join(n["title"] for n in _notes()) or "(no notes)"


@server.tool()
def search_notes(query: str) -> str:
    """Full-text search over the team notes; returns matching notes (title and body)."""
    q = query.lower().split()
    hits = [n for n in _notes() if all(w in (n["title"] + " " + n["body"]).lower() for w in q)]
    return clip("\n\n".join(f"# {n['title']}\n{n['body']}" for n in hits)) if hits else "no matching notes"


if __name__ == "__main__":
    server.run(transport="stdio")
