from __future__ import annotations

import asyncio
from typing import Any

from .config import VoicePoCConfig
from .mcp_client import MCPClient
from .runtime import SpeechRuntime


class VoiceChatPipeline:
    def __init__(self, config: VoicePoCConfig, runtime: SpeechRuntime):
        self.config = config
        self.runtime = runtime
        self.session_id: str | None = None
        self.client = MCPClient(config.mcp_url, config.mcp_api_key)

    def run_sync(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        print("mojo-voice PoC started. Type 'exit' to quit.")
        await self.client.initialize()

        while True:
            text = await self.runtime.transcribe_from_mic()
            if not text:
                continue
            if text.lower() in {"exit", "quit", "q"}:
                print("bye")
                return

            reply = await self._send_to_dialog(text)
            await self.runtime.speak(reply)

    async def _send_to_dialog(self, message: str) -> str:
        return await self._dialog_chat(message)

    async def _dialog_chat(self, message: str) -> str:
        args: dict[str, Any] = {
            "action": "chat",
            "role_id": self.config.mcp_brain_role,
            "message": message,
        }
        if self.session_id:
            args["session_id"] = self.session_id

        result = await self.client.call_tool("dialog", args)
        payload = result.tool_payload()
        if payload.get("session_id"):
            self.session_id = str(payload["session_id"])

        response = payload.get("response")
        if isinstance(response, str) and response.strip():
            return response.strip()

        msg = payload.get("message") or payload.get("error") or "(No response)"
        return str(msg)
