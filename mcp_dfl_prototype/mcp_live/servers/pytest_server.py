import sys, subprocess
from mcp.server.mcpserver import MCPServer
from _common import WS, clip

server = MCPServer("pytest", instructions="Runs the project's test suite and reports the results.")


@server.tool()
def run_tests(args: str = "") -> str:
    """Run `pytest` in the workspace (optionally with extra arguments, e.g. a path or -k expr); returns the summary."""
    cmd = [sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider", *args.split()]
    try:
        r = subprocess.run(cmd, cwd=WS, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return "error: pytest timed out"
    lines = (r.stdout + r.stderr).strip().splitlines()
    return clip("\n".join(lines[-40:]))


if __name__ == "__main__":
    server.run(transport="stdio")
