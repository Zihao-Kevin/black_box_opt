from mcp.server.mcpserver import MCPServer
from _common import append_json

server = MCPServer("mail", instructions="Sends e-mail from the agent's company mailbox.")


@server.tool()
def send_mail(to: str, subject: str, body: str) -> str:
    """Send an e-mail to the address `to` with the given subject and body."""
    n = append_json(".outbox.json", {"to": to, "subject": subject, "body": body})
    return f"sent message #{n} to {to}"


if __name__ == "__main__":
    server.run(transport="stdio")
