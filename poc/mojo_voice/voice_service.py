from __future__ import annotations

import json
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

_CONDUCTOR_SYSTEM = """\
You are a voice conversation conductor. Your job: keep the conversation alive and help the user while a research task runs in the background.

Each turn, pick exactly one move and respond in 1-2 natural spoken sentences:
- CLARIFY: ask the single most important missing question
- OPTIONS: offer 2-3 concrete options the user can choose from
- ACKNOWLEDGE: confirm you received and are working on it
- BRIDGE: weave in the pending_mcp_result naturally into speech
- STALL: let the user know it is still processing, keep them engaged

Rules:
- Never more than 2 sentences
- No markdown, no lists, spoken language only
- If pending_mcp_result is provided → BRIDGE takes priority
- If clarity_score < 0.6 → prefer CLARIFY or OPTIONS
- If text_brain_status is complete and no pending result → BRIDGE last known result
"""


@dataclass
class VoiceTurnResult:
    transcript: str
    reply_text: str
    reply_audio_base64: str
    session_id: str


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
    # Main turn — two-brain flow                                          #
    # ------------------------------------------------------------------ #

    async def process_turn(self, audio_base64: str, session_id: str) -> VoiceTurnResult:
        session = store.get(session_id)
        if session is None:
            raise RuntimeError(f"Session not found: {session_id}")

        # STT
        audio_bytes = decode_audio_base64(audio_base64)
        wav16k = to_wav16k_mono_bytes(audio_bytes)
        wav_path = write_temp_wav(wav16k)
        try:
            stt = self.stt.transcribe_wav_path(wav_path)
        finally:
            safe_unlink(wav_path)

        transcript = stt.text.strip()
        session.touch()

        if not session.original_ask:
            # First turn: record intent, estimate clarity, dispatch MCP brain
            session.original_ask = transcript
            session.clarity_score = self._estimate_clarity(transcript)
            await self._dispatch_mcp_brain(session)
        else:
            # Subsequent turns: push transcript as context for MCP brain
            update = ContextUpdate(
                id=str(uuid.uuid4()),
                type="clarification",
                content=f"User said: {transcript}",
            )
            session.context_queue.append(update)
            session.context_version += 1

        # Poll MCP brain for progress / results
        await self._poll_mcp_brain(session)

        # Conductor picks a move and replies
        reply_text = await self._run_conductor(transcript, session)
        if not reply_text.strip():
            raise RuntimeError("Conductor returned empty reply")

        wav_out = self.tts.synthesize_wav(reply_text)
        return VoiceTurnResult(
            transcript=transcript,
            reply_text=reply_text,
            reply_audio_base64=encode_audio_base64(wav_out),
            session_id=session.session_id,
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

    # ------------------------------------------------------------------ #
    # MCP brain — dispatch and poll                                       #
    # ------------------------------------------------------------------ #

    async def _dispatch_mcp_brain(self, session: VoiceSession) -> None:
        result = await self.client.call_tool(
            "scheduler",
            {
                "action": "add",
                "role_id": self.cfg.mcp_brain_role,
                "goal": session.original_ask,
                "task_type": "agentic",
            },
        )
        payload = result.tool_payload()
        task_id = payload.get("task_id")
        if not task_id:
            raise RuntimeError(f"Scheduler did not return task_id: {payload}")
        session.text_brain_task_id = task_id
        session.text_brain_status = "running"
        logger.info(
            "MCP brain dispatched task_id=%s for session %s",
            task_id, session.session_id,
        )

    async def _poll_mcp_brain(self, session: VoiceSession) -> None:
        if not session.text_brain_task_id or session.text_brain_status == "complete":
            return
        result = await self.client.call_tool(
            "scheduler",
            {
                "action": "get",
                "task_id": session.text_brain_task_id,
            },
        )
        payload = result.tool_payload()
        task_status = payload.get("status", "")

        if task_status in ("completed", "completed_auto_extracted"):
            session.text_brain_status = "complete"
            summary = (
                payload.get("result")
                or payload.get("reply")
                or payload.get("goal", "")
            )
            if summary:
                wav_bytes = self.tts.synthesize_wav(str(summary))
                item = AudioQueueItem(
                    id=str(uuid.uuid4()),
                    type="result",
                    summary=str(summary),
                    reply_audio_base64=encode_audio_base64(wav_bytes),
                )
                session.audio_queue.append(item)
                logger.info("MCP brain result queued for session %s", session.session_id)

        elif task_status == "failed":
            session.text_brain_status = "complete"
            raise RuntimeError(
                f"MCP brain task {session.text_brain_task_id} failed: "
                f"{payload.get('error') or payload}"
            )

        elif task_status == "running":
            progress = payload.get("progress") or payload.get("working_on", "")
            if progress:
                session.text_brain_working_on = str(progress)

    # ------------------------------------------------------------------ #
    # Conductor — fast 5-move classifier                                  #
    # ------------------------------------------------------------------ #

    async def _run_conductor(self, transcript: str, session: VoiceSession) -> str:
        pending = session.next_pending_audio()
        pending_mcp_result = None
        if pending:
            pending_mcp_result = {"type": pending.type, "summary": pending.summary}
            session.mark_audio_played(pending.id)

        user_message = json.dumps(
            {
                "transcript": transcript,
                "session_context": {
                    "original_ask": session.original_ask,
                    "clarity_score": session.clarity_score,
                    "missing_context": [
                        m.field for m in session.missing_context if not m.answered
                    ],
                    "pending_mcp_result": pending_mcp_result,
                    "text_brain_status": session.text_brain_status,
                },
            },
            ensure_ascii=False,
        )

        resource_id = os.getenv("AUDIO_BRAIN_RESOURCE_ID", "lmstudio")
        result = await self.client.call_tool(
            "llm_direct_chat",
            {
                "system_prompt": _CONDUCTOR_SYSTEM,
                "message": user_message,
                "resource_id": resource_id,
                "max_tokens": 128,
            },
        )
        payload = result.tool_payload()
        if payload.get("status") == "error":
            raise RuntimeError(f"Conductor LLM error: {payload.get('message')}")

        reply = (payload.get("reply") or "").strip()
        resource = payload.get("resource_id", "")
        model = payload.get("model", "")
        logger.info(
            "Conductor reply via %s (%s): %s chars", resource, model, len(reply)
        )
        return reply

    # ------------------------------------------------------------------ #
    # Clarity heuristic                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _estimate_clarity(text: str) -> float:
        words = len(text.split())
        if words < 4:
            return 0.3
        if words < 10:
            return 0.6
        return 0.85

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
