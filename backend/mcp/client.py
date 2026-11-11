# backend/mcp/client.py
# Minimal JSON-RPC client for the stateless MCP servers (knowledge base, web search).

import json
from typing import Any

import httpx

from backend.core.logger import get_logger

logger = get_logger(__name__)


async def call_mcp_tool(
    server_url: str,
    tool_name: str,
    arguments: dict[str, Any],
    timeout: float = 30.0,
) -> Any:
    """Call one tool on a stateless MCP server (stateless_http=True: each POST stands alone, no
    initialize handshake). server_url is the base URL, e.g. "http://localhost:8010/mcp/kb";
    tool_name matches the @mcp.tool() name. Returns the tool result decoded from JSON.

    Raises httpx.HTTPStatusError (4xx/5xx), ValueError (JSON-RPC error from the tool) and
    httpx.TimeoutException.
    """
    payload = {
        "jsonrpc": "2.0",
        "id":      1,
        "method":  "tools/call",
        "params": {
            "name":      tool_name,
            "arguments": arguments,
        },
    }

    # Servers with json_response=True require Accept: application/json, or they answer
    # -32600 "Not Acceptable"
    headers = {
        "Content-Type": "application/json",
        "Accept":        "application/json",
    }

    # trust_env=False ignores HTTP_PROXY / ALL_PROXY and system proxies. Some proxies (macOS
    # settings, debugging tools) do not exclude localhost and turn local MCP calls into 502s;
    # internal calls never need a proxy.
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        resp = await client.post(f"{server_url}/mcp", json=payload, headers=headers)
        resp.raise_for_status()

    data = resp.json()

    # JSON-RPC error envelope
    if "error" in data:
        raise ValueError(
            f"MCP tool '{tool_name}' error: {data['error'].get('message', str(data['error']))}"
        )

    # FastMCP serializes list[dict] either as one TextContent per dict (most versions) or as
    # one TextContent holding the whole list. Read every entry, not just content[0], or only the
    # first result comes back.
    content = data.get("result", {}).get("content", [])
    if not content:
        return []

    items = []
    for item in content:
        if not isinstance(item, dict):
            continue
        text = item.get("text", "")
        if not text:
            continue
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            parsed = text
        if isinstance(parsed, list):
            # The whole list in one TextContent
            return parsed
        items.append(parsed)

    return items if items else content


async def list_mcp_tools(server_url: str, timeout: float = 10.0) -> list[dict]:
    """List the server's tools (name / description / inputSchema), for debugging."""
    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        resp = await client.post(f"{server_url}/mcp", json=payload, headers=headers)
        resp.raise_for_status()

    return resp.json().get("result", {}).get("tools", [])
