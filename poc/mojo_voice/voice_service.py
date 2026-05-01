from __future__ import annotations

import logging
import os
import uuid
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
from .session_store import AudioQueueItem, ContextUpdate, VoiceSession, store
from .stt_funasr import FunASRSTT
from .stt_base import STTEngine
from .stt_zai import ZaiASRSTT
from .tts_cosyvoice2 import CosyVoiceTTS

logger = logging.getLogger(__name__)

# Audio brain system prompt — injected on every /voice/query turn.
_AUDIO_BRAIN_SYSTEM = """\
You are the audio co-pilot in a two-brain assistant system.
A powerful text engine is working on the user's request in the background.

Your parallel responsibilities each turn:
1. CLARITY — score the user's ask (0.0–1.0). If below 0.7, identify the single
   most important missing context and ask ONE focused probing question.
   Never ask more than one question at a time.
2. CONTEXT — when the user answers a probing question, note the clarification
   and indicate it should update the text engine.
3. PRESENT — if there are pending results from the text engine, weave the key
   point into your reply naturally in 1–2 spoken sentences.
4. ENGAGE — keep the user engaged and thinking. Never be idle.

Hard rules:
- Reply in natural spoken language only. No markdown, no lists, no tables.
- Maximum 3 sentences per reply. Be concise and expressive.
- If the user's request is clear and no results are pending, acknowledge briefly
  and let them know you are working on it.
"""


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

        self.stt: STTEngine = self._build_stt_engine(cfg.stt_provider)

        cosy_model_path = os.getenv(
            "COSYVOICE_MODEL_PATH",
            os.getenv("COSYVOICE2_MODEL_PATH", "pretrained_models/Fun-CosyVoice3-0.5B"),
        )
        cosy_speaker = os.getenv(
            "COSYVOICE_SPEAKER",
            os.getenv("COSYVOICE2_SPEAKER", ""),
        )
        self.tts = CosyVoiceTTS(model_path=cosy_model_path, speaker=cosy_speaker)

    def _build_stt_engine(self, provider: str) -> STTEngine:
        p = (provider or "").strip().lower()
        if p == "funasr":
            funasr_model = os.getenv("FUNASR_MODEL", "iic/SenseVoiceSmall")
            funasr_vad = os.getenv("FUNASR_VAD_MODEL", "fsmn-vad")
            funasr_punc = os.getenv("FUNASR_PUNC_MODEL", "ct-punc")
            funasr_device = os.getenv("FUNASR_DEVICE", "cpu")
            return FunASRSTT(
                model=funasr_model,
                vad_model=funasr_vad,
                punc_model=funasr_punc,
                device=funasr_device,
            )
        if p == "zai_asr":
            return ZaiASRSTT.from_env()
        raise RuntimeError(
            f"Unsupported MOJO_STT_PROVIDER={provider!r}. "
            "Supported providers: funasr, zai_asr"
        )

    async def initialize(self) -> None:
        await self.client.initialize()
        self.stt.load()
        self.tts.load()

    # ------------------------------------------------------------------ #
    # Core audio query — session-aware                                    #
    # ------------------------------------------------------------------ #

    async def query_audio_base64(
        self,
        audio_base64: str,
        session_id: str | None = None,
        mode: str | None = None,
        role_id: str | None = None,
    ) -> VoiceQueryResult:
        session = store.get(session_id) if session_id else None

        audio_bytes = decode_audio_base64(audio_base64)
        wav16k = to_wav16k_mono_bytes(audio_bytes)
        wav_path = write_temp_wav(wav16k)
        try:
            stt = self.stt.transcribe_wav_path(wav_path)
        finally:
            safe_unlink(wav_path)

        transcript = stt.text.strip()

        if session:
            session.touch()
            if not session.original_ask:
                session.original_ask = transcript

        selected_mode = (mode or self.cfg.mcp_mode or "").strip().lower()
        # If caller explicitly asks for MCP-backed answer path, route directly.
        if selected_mode in ("search_memory", "dialog"):
            reply_text = await self._ask_mcp(transcript, selected_mode, role_id)
        else:
            prompt = self._build_audio_brain_prompt(transcript, session)
            reply_text = await self._ask_audio_brain(prompt)
            # Fallback: if audio brain returns placeholder/empty, fetch real MCP answer.
            if not reply_text.strip() or reply_text.strip().lower() in (
                "i'm still processing your request.",
                "i'm still thinking, give me a moment.",
            ):
                reply_text = await self._ask_mcp(
                    transcript,
                    self.cfg.mcp_mode or "search_memory",
                    role_id or self.cfg.role_id,
                )

        # Push any clarification the audio brain produced back to the context queue
        if session:
            self._extract_context_update(session, transcript, reply_text)

        wav_out = self.tts.synthesize_wav(reply_text)
        return VoiceQueryResult(
            transcript=transcript,
            reply_text=reply_text,
            reply_audio_base64=encode_audio_base64(wav_out),
            session_id=session.session_id if session else session_id,
        )

    def transcribe_audio_base64(self, audio_base64: str) -> str:
        audio_bytes = decode_audio_base64(audio_base64)
        wav16k = to_wav16k_mono_bytes(audio_bytes)
        wav_path = write_temp_wav(wav16k)
        try:
            stt = self.stt.transcribe_wav_path(wav_path)
        finally:
            safe_unlink(wav_path)
        transcript = stt.text.strip()
        if not transcript:
            raise RuntimeError("Empty transcript")
        return transcript

    def _build_audio_brain_prompt(
        self, transcript: str, session: VoiceSession | None
    ) -> str:
        parts = [_AUDIO_BRAIN_SYSTEM]

        if session:
            pending = session.next_pending_audio()
            if pending:
                parts.append(
                    f"\n[Pending result from text engine — type: {pending.type}]\n"
                    f"{pending.summary}\n"
                    f"Present this naturally in your reply, then mark it as delivered.\n"
                )
                session.mark_audio_played(pending.id)

            if session.refined_ask and session.refined_ask != session.original_ask:
                parts.append(f"\n[Refined ask so far]: {session.refined_ask}")

            gaps = [m for m in session.missing_context if not m.asked]
            if gaps:
                parts.append(
                    f"\n[Missing context not yet asked]: "
                    + ", ".join(g.field for g in gaps)
                )

        parts.append(f"\nUser said: {transcript}")
        return "\n".join(parts)

    def _extract_context_update(
        self, session: VoiceSession, transcript: str, reply_text: str
    ) -> None:
        """
        Heuristic: if the audio brain asked a question, record the user's
        transcript as a context clarification for the text brain.
        The text brain polls /voice/context to consume these.
        """
        if "?" in reply_text:
            # Audio brain asked something — next user turn will be the answer.
            # For now push this turn's transcript as context signal.
            update = ContextUpdate(
                id=str(uuid.uuid4()),
                type="clarification",
                content=f"User said: {transcript}",
            )
            session.context_queue.append(update)
            session.context_version += 1

    # ------------------------------------------------------------------ #
    # Push endpoint — text brain → audio queue                           #
    # ------------------------------------------------------------------ #

    async def push_to_session(
        self, session_id: str, summary: str, push_type: str
    ) -> bool:
        session = store.get(session_id)
        if session is None:
            return False
        session.touch()

        wav_bytes = self.tts.synthesize_wav(summary)
        item = AudioQueueItem(
            id=str(uuid.uuid4()),
            type=push_type,
            summary=summary,
            reply_audio_base64=encode_audio_base64(wav_bytes),
        )
        session.audio_queue.append(item)
        logger.info("Queued %s audio item for session %s", push_type, session_id)
        return True

    # ------------------------------------------------------------------ #
    # Poll endpoints                                                      #
    # ------------------------------------------------------------------ #

    def pop_pending_audio(self, session_id: str) -> AudioQueueItem | None:
        session = store.get(session_id)
        if session is None:
            return None
        session.touch()
        item = session.next_pending_audio()
        if item:
            session.mark_audio_played(item.id)
        return item

    def pop_pending_context(self, session_id: str) -> ContextUpdate | None:
        session = store.get(session_id)
        if session is None:
            return None
        session.touch()
        item = session.next_pending_context()
        if item:
            session.mark_context_consumed(item.id)
        return item

    # ------------------------------------------------------------------ #
    # Audio brain — routes through fastest free resource via MoJo pool  #
    # ------------------------------------------------------------------ #

    async def _ask_audio_brain(self, user_message: str) -> str:
        """
        Single-turn LLM call for the audio brain conductor.
        Uses llm_direct_chat which picks the fastest benchmarked free resource.
        The system prompt is passed separately so the model treats it correctly.
        """
        # Split the built prompt: first section is system, last line is user message
        # _build_audio_brain_prompt appends "\nUser said: {transcript}" at the end.
        # We pass the conductor instructions as system_prompt and the user turn separately.
        lines = user_message.rsplit("\nUser said: ", 1)
        if len(lines) == 2:
            system_part = lines[0].strip()
            user_part = lines[1].strip()
        else:
            system_part = _AUDIO_BRAIN_SYSTEM.strip()
            user_part = user_message.strip()

        resource_id = os.getenv("AUDIO_BRAIN_RESOURCE_ID", "lmstudio")
        result = await self.client.call_tool(
            "llm_direct_chat",
            {
                "system_prompt": system_part,
                "message": user_part,
                "resource_id": resource_id,
                "max_tokens": 256,
            },
        )
        payload = result.tool_payload()
        if payload.get("status") == "error":
            logger.warning("llm_direct_chat error: %s", payload.get("message"))
            return "I'm still thinking, give me a moment."

        reply = (payload.get("reply") or "").strip()
        resource = payload.get("resource_id", "")
        model = payload.get("model", "")
        logger.info("Audio brain reply via %s (%s): %s chars", resource, model, len(reply))
        return reply or "I'm still processing your request."

    # ------------------------------------------------------------------ #
    # MCP helpers — used for memory/dialog fallback and push synthesis   #
    # ------------------------------------------------------------------ #

    async def _ask_mcp(
        self, message: str, mode: str | None, role_id: str | None
    ) -> str:
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
        result = await self.client.call_tool("dialog", args)
        payload = result.tool_payload()
        response = payload.get("response")
        if isinstance(response, str) and response.strip():
            return response.strip()
        return str(payload.get("message") or payload.get("error") or "(No response)")
