import os
from mcp.server.mcpserver import MCPServer
from _common import safe

server = MCPServer("write_file", instructions="Creates or overwrites a text file in the project workspace.")


@server.tool()
def write_file(path: str, content: str) -> str:
    """Write `content` to the workspace file at the relative `path` (parent directories are created)."""
    p = safe(path)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(content)
    return f"wrote {len(content)} characters to {path}"


if __name__ == "__main__":
    server.run(transport="stdio")
