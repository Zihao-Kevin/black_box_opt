from mcp.server.mcpserver import MCPServer
from _common import append_json

server = MCPServer("ticket", instructions="Creates tickets in the team's issue tracker.")


@server.tool()
def create_ticket(title: str, body: str = "") -> str:
    """Open a new ticket in the issue tracker with the given title and description."""
    n = append_json(".tickets.json", {"title": title, "body": body})
    return f"created ticket #{n}: {title}"


if __name__ == "__main__":
    server.run(transport="stdio")
