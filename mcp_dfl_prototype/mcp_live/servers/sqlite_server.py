import os, sqlite3
from mcp.server.mcpserver import MCPServer
from _common import WS, clip

server = MCPServer("sqlite", instructions="Runs read-only SQL queries against the project database data.db.")
DB = os.path.join(WS, "data.db")


def _conn():
    c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    c.execute("PRAGMA query_only = 1")
    return c


@server.tool()
def list_tables() -> str:
    """List the tables of the database with their CREATE statements."""
    rows = _conn().execute("SELECT sql FROM sqlite_master WHERE type='table'").fetchall()
    return "\n".join(r[0] for r in rows)


@server.tool()
def query(sql: str, max_rows: int = 50) -> str:
    """Run a read-only SQL SELECT and return the rows as text."""
    cur = _conn().execute(sql)
    cols = [d[0] for d in cur.description] if cur.description else []
    rows = cur.fetchmany(max_rows)
    lines = [" | ".join(cols)] + [" | ".join(str(v) for v in r) for r in rows]
    if len(rows) == max_rows:
        lines.append(f"[first {max_rows} rows]")
    return clip("\n".join(lines))


if __name__ == "__main__":
    server.run(transport="stdio")
