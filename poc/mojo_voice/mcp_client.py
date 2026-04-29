from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class MCPToolResponse:
    raw: dict[str, Any]

    def result_json(self) -> dict[str, Any]:
        return self.raw.get("result", {}) if isinstance(self.raw, dict) else {}

    def tool_payload(self) -> dict[str, Any]:
        """
        MoJo tools/call response shape:
          result.content[0].text => JSON string payload.
        """
        result = self.result_json()
        content = result.get("content", [])
        if not content:
            return {}
        text = (content[0] or {}).get("text", "")
        if not text:
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"status": "error", "message": f"Invalid tool payload: {text[:200]}"}


class MCPClient:
    def __init__(self, base_url: str, api_key: str | None = None, timeout_s: float = 30.0):
        self.base_url = base_url.rstrip("/") + "/"
        self.api_key = api_key
        self.timeout = timeout_s
        self._id = 1

    async def initialize(self) -> None:
        await self._request("initialize", {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "mojo-voice-poc", "version": "0.1.0"}})

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> MCPToolResponse:
        raw = await self._request("tools/call", {"name": tool_name, "arguments": arguments})
        return MCPToolResponse(raw=raw)

    async def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["MCP-API-Key"] = self.api_key

        body = {
            "jsonrpc": "2.0",
            "id": self._id,
            "method": method,
            "params": params,
        }
        self._id += 1

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(self.base_url, json=body, headers=headers)
            resp.raise_for_status()
            payload = resp.json()
            if "error" in payload:
                raise RuntimeError(f"MCP error: {payload['error']}")
            return payload
