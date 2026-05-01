from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass
class VoicePoCConfig:
    mcp_url: str
    mcp_api_key: str | None
    role_id: str
    voice_runtime: str
    mcp_mode: str
    stt_provider: str


def load_config() -> VoicePoCConfig:
    load_dotenv()
    runtime = os.getenv("MOJO_VOICE_RUNTIME", "mock").strip().lower()
    if runtime not in {"mock", "executorch"}:
        runtime = "mock"
    mcp_mode = os.getenv("MOJO_MCP_MODE", "search_memory").strip().lower()
    if mcp_mode not in {"search_memory", "dialog"}:
        mcp_mode = "search_memory"

    api_key = (
        os.getenv("MOJO_MCP_API_KEY", "").strip()
        or os.getenv("MCP_API_KEY", "").strip()
        or None
    )
    return VoicePoCConfig(
        mcp_url=os.getenv("MOJO_MCP_URL", "http://127.0.0.1:8000/").strip(),
        mcp_api_key=api_key,
        role_id=os.getenv("MOJO_ROLE_ID", "assistant").strip(),
        voice_runtime=runtime,
        mcp_mode=mcp_mode,
        stt_provider=os.getenv("MOJO_STT_PROVIDER", "funasr").strip().lower(),
    )
