import sys, subprocess
from mcp.server.mcpserver import MCPServer
from _common import WS, safe, clip

server = MCPServer("lint", instructions="Runs the static linter (pyflakes) on a workspace file and reports every warning.")


@server.tool()
def lint(path: str) -> str:
    """Lint one Python file (relative path) with pyflakes; returns the warnings and their count."""
    r = subprocess.run([sys.executable, "-m", "pyflakes", safe(path)], cwd=WS, capture_output=True, text=True, timeout=30)
    lines = [l.replace(WS + "/", "") for l in (r.stdout + r.stderr).strip().splitlines() if l.strip()]
    return clip("\n".join(lines) + f"\n{len(lines)} warning(s)") if lines else "0 warning(s)"


if __name__ == "__main__":
    server.run(transport="stdio")
