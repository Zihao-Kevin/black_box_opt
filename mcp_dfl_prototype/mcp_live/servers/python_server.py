import sys, subprocess, tempfile
from mcp.server.mcpserver import MCPServer
from _common import clip

server = MCPServer("python", instructions="Executes a Python snippet in an isolated sandbox (no access to the workspace files) and returns its output.")


@server.tool()
def run_python(code: str) -> str:
    """Run Python code in a fresh interpreter (10 s limit); returns stdout and stderr. The sandbox cannot see workspace files: pass data in the code itself."""
    with tempfile.TemporaryDirectory() as d:
        try:
            r = subprocess.run([sys.executable, "-I", "-c", code], cwd=d, capture_output=True, text=True, timeout=10)
        except subprocess.TimeoutExpired:
            return "error: timed out after 10 s"
    return clip((r.stdout + (("\nstderr:\n" + r.stderr) if r.stderr else "")).strip() or "(no output)")


if __name__ == "__main__":
    server.run(transport="stdio")
