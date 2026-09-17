from mcp.server.mcpserver import MCPServer
from _common import safe, visible_files, clip

server = MCPServer("filesystem", instructions="Lists and reads the files of the project workspace.")


@server.tool()
def list_files() -> str:
    """List every file in the workspace (relative paths)."""
    return "\n".join(visible_files())


@server.tool()
def read_file(path: str) -> str:
    """Read a text file from the workspace by relative path."""
    return clip(open(safe(path), encoding="utf-8", errors="replace").read())


if __name__ == "__main__":
    server.run(transport="stdio")
