import re, urllib.request
from mcp.server.mcpserver import MCPServer
from _common import clip

server = MCPServer("fetch", instructions="Fetches a web page by URL (the company intranet) and returns its text.")


@server.tool()
def fetch(url: str) -> str:
    """HTTP GET a URL and return the page as plain text (HTML tags stripped)."""
    if not url.startswith(("http://", "https://")):
        raise ValueError("only http(s) URLs")
    with urllib.request.urlopen(url, timeout=10) as r:
        body = r.read().decode("utf-8", errors="replace")
    text = re.sub(r"<script.*?</script>|<style.*?</style>", "", body, flags=re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    return clip(re.sub(r"[ \t]+", " ", re.sub(r"\n\s*\n+", "\n", text)).strip())


if __name__ == "__main__":
    server.run(transport="stdio")
