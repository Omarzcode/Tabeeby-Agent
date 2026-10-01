"""A tiny external MCP server used to test Tabeeby's MCP *client* side."""
from mcp.server import MCPServer

mcp = MCPServer("echo")


@mcp.tool()
def shout(text: str) -> str:
    """Return the text in upper case."""
    return text.upper()


@mcp.tool()
def fail(reason: str) -> str:
    """Always fails."""
    raise ValueError(reason)


if __name__ == "__main__":
    mcp.run()
