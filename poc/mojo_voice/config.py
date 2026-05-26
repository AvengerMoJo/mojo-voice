from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass
class VoicePoCConfig:
    mcp_url: str
    mcp_api_key: str | None
    voice_runtime: str
    stt_provider: str
    mcp_brain_role: str


def load_config() -> VoicePoCConfig:
    load_dotenv()

    runtime = os.getenv("MOJO_VOICE_RUNTIME", "mock").strip().lower()
    if runtime not in {"mock", "executorch"}:
        raise RuntimeError(
            f"Invalid MOJO_VOICE_RUNTIME={runtime!r}. Supported: mock, executorch"
        )

    api_key = (
        os.getenv("MOJO_MCP_API_KEY", "").strip()
        or os.getenv("MCP_API_KEY", "").strip()
        or None
    )
    return VoicePoCConfig(
        mcp_url=os.getenv("MOJO_MCP_URL", "http://127.0.0.1:8000/").strip(),
        mcp_api_key=api_key,
        voice_runtime=runtime,
        stt_provider=os.getenv("MOJO_STT_PROVIDER", "funasr").strip().lower(),
        mcp_brain_role=os.getenv("MOJO_MCP_BRAIN_ROLE", "popo").strip(),
    )
