from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from .audio_utils import (
    decode_audio_base64,
    encode_audio_base64,
    safe_unlink,
    to_wav16k_mono_bytes,
    write_temp_wav,
)
from .config import VoicePoCConfig
from .mcp_client import MCPClient
from .stt_funasr import FunASRSTT
from .tts_cosyvoice2 import CosyVoiceTTS

logger = logging.getLogger(__name__)


@dataclass
class VoiceQueryResult:
    transcript: str
    reply_text: str
    reply_audio_base64: str
    session_id: str | None


class VoiceService:
    def __init__(self, cfg: VoicePoCConfig):
        self.cfg = cfg
        self.client = MCPClient(cfg.mcp_url, cfg.mcp_api_key)
        self.session_id: str | None = None

        funasr_model = os.getenv("FUNASR_MODEL", "iic/SenseVoiceSmall")
        funasr_vad = os.getenv("FUNASR_VAD_MODEL", "fsmn-vad")
        funasr_punc = os.getenv("FUNASR_PUNC_MODEL", "ct-punc")
        funasr_device = os.getenv("FUNASR_DEVICE", "cpu")
        self.stt = FunASRSTT(
            model=funasr_model,
            vad_model=funasr_vad,
            punc_model=funasr_punc,
            device=funasr_device,
        )

        cosy_model_path = os.getenv(
            "COSYVOICE_MODEL_PATH",
            os.getenv("COSYVOICE2_MODEL_PATH", "pretrained_models/Fun-CosyVoice3-0.5B"),
        )
        cosy_speaker = os.getenv(
            "COSYVOICE_SPEAKER",
            os.getenv("COSYVOICE2_SPEAKER", ""),
        )
        self.tts = CosyVoiceTTS(model_path=cosy_model_path, speaker=cosy_speaker)

    async def initialize(self) -> None:
        await self.client.initialize()
        self.stt.load()
        self.tts.load()

    async def query_audio_base64(self, audio_base64: str, mode: str | None = None, role_id: str | None = None) -> VoiceQueryResult:
        audio_bytes = decode_audio_base64(audio_base64)
        wav16k = to_wav16k_mono_bytes(audio_bytes)
        wav_path = write_temp_wav(wav16k)

        try:
            stt = self.stt.transcribe_wav_path(wav_path)
        finally:
            safe_unlink(wav_path)

        transcript = stt.text.strip()
        if not transcript:
            transcript = ""

        reply_text = await self._ask_mcp(transcript, mode=mode, role_id=role_id)
        wav_out = self.tts.synthesize_wav(reply_text)
        return VoiceQueryResult(
            transcript=transcript,
            reply_text=reply_text,
            reply_audio_base64=encode_audio_base64(wav_out),
            session_id=self.session_id,
        )

    async def _ask_mcp(self, message: str, mode: str | None, role_id: str | None) -> str:
        selected_mode = (mode or self.cfg.mcp_mode or "search_memory").strip().lower()
        if selected_mode == "dialog":
            return await self._mcp_dialog(message, role_id or self.cfg.role_id)
        return await self._mcp_search_memory(message)

    async def _mcp_search_memory(self, message: str) -> str:
        result = await self.client.call_tool(
            "search_memory",
            {
                "query": message,
                "types": ["conversations", "documents"],
                "limit_per_type": 3,
                "max_content_chars": 220,
            },
        )
        payload = result.tool_payload()
        if payload.get("status") == "error":
            return str(payload.get("message") or "search_memory failed")

        buckets = payload.get("results", {})
        lines: list[str] = []
        for bucket in ("conversations", "documents"):
            hits = buckets.get(bucket) or []
            for idx, hit in enumerate(hits, start=1):
                content = str(hit.get("content") or "").strip()
                if content:
                    lines.append(f"{bucket[:-1]} {idx}: {content}")
        return "\n".join(lines[:4]) if lines else "I could not find relevant memory results."

    async def _mcp_dialog(self, message: str, role_id: str) -> str:
        args: dict[str, str] = {
            "action": "chat",
            "role_id": role_id,
            "message": message,
        }
        if self.session_id:
            args["session_id"] = self.session_id

        result = await self.client.call_tool("dialog", args)
        payload = result.tool_payload()
        sid = payload.get("session_id")
        if sid:
            self.session_id = str(sid)

        response = payload.get("response")
        if isinstance(response, str) and response.strip():
            return response.strip()
        return str(payload.get("message") or payload.get("error") or "(No response)")
