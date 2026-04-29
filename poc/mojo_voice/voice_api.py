from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import load_config
from .voice_service import VoiceService

logger = logging.getLogger(__name__)


class VoiceQueryRequest(BaseModel):
    audio_base64: str = Field(..., description="Base64 audio payload; data URI prefix is allowed")
    mcp_mode: str | None = Field(default=None, description="search_memory or dialog")
    role_id: str | None = Field(default=None)


class VoiceQueryResponse(BaseModel):
    transcript: str
    reply_text: str
    reply_audio_base64: str
    reply_audio_format: str = "wav"
    session_id: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    service = VoiceService(cfg)
    try:
        await service.initialize()
    except Exception as exc:
        logger.warning("Voice service warmup failed: %s", exc)
    app.state.voice_service = service
    yield


app = FastAPI(title="mojo-voice API", version="0.2.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/voice/query", response_model=VoiceQueryResponse)
async def voice_query(req: VoiceQueryRequest):
    service: VoiceService = app.state.voice_service
    try:
        res = await service.query_audio_base64(
            audio_base64=req.audio_base64,
            mode=req.mcp_mode,
            role_id=req.role_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return VoiceQueryResponse(
        transcript=res.transcript,
        reply_text=res.reply_text,
        reply_audio_base64=res.reply_audio_base64,
        session_id=res.session_id,
    )
