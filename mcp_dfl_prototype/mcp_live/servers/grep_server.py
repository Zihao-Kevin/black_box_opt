import re, os
from mcp.server.mcpserver import MCPServer
from _common import WS, visible_files, clip

server = MCPServer("grep", instructions="Searches the text of every workspace file for a regular expression.")


@server.tool()
def search(pattern: str, max_hits: int = 30) -> str:
    """Search all workspace files for a regex; returns `path:line: text` for each matching line."""
    rx = re.compile(pattern)
    hits = []
    for rel in visible_files():
        try:
            for i, line in enumerate(open(os.path.join(WS, rel), encoding="utf-8", errors="replace"), 1):
                if rx.search(line):
                    hits.append(f"{rel}:{i}: {line.rstrip()}")
                    if len(hits) >= max_hits:
                        return clip("\n".join(hits) + f"\n[stopped at {max_hits} hits]")
        except (UnicodeDecodeError, OSError):
            continue
    return clip("\n".join(hits)) if hits else "no matches"


if __name__ == "__main__":
    server.run(transport="stdio")
