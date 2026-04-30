from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import load_config
from .session_store import store
from .voice_service import VoiceService

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Request / response models                                           #
# ------------------------------------------------------------------ #

class VoiceQueryRequest(BaseModel):
    audio_base64: str = Field(..., description="Base64 WAV; data URI prefix allowed")
    mcp_mode: str | None = Field(default=None, description="search_memory or dialog")
    role_id: str | None = Field(default=None)


class VoiceQueryResponse(BaseModel):
    transcript: str
    reply_text: str
    reply_audio_base64: str
    reply_audio_format: str = "wav"
    session_id: str | None = None


class SessionCreateResponse(BaseModel):
    session_id: str


class PushRequest(BaseModel):
    type: str = Field(..., description="progress | result | question")
    summary: str = Field(..., description="2-3 sentence plain text, no markdown")


class PushResponse(BaseModel):
    queued: bool


class PendingAudioResponse(BaseModel):
    pending: bool
    type: str | None = None
    reply_text: str | None = None
    reply_audio_base64: str | None = None
    reply_audio_format: str = "wav"


class PendingContextResponse(BaseModel):
    update: bool
    type: str | None = None
    content: str | None = None
    context_version: int | None = None


class SessionStateResponse(BaseModel):
    session_id: str
    intent: dict
    text_brain: dict


# ------------------------------------------------------------------ #
# App lifecycle                                                        #
# ------------------------------------------------------------------ #

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


app = FastAPI(title="mojo-voice API", version="0.3.0", lifespan=lifespan)


# ------------------------------------------------------------------ #
# Health                                                               #
# ------------------------------------------------------------------ #

@app.get("/health")
async def health():
    return {"status": "ok"}


# ------------------------------------------------------------------ #
# Session lifecycle                                                    #
# ------------------------------------------------------------------ #

@app.post("/voice/session", response_model=SessionCreateResponse, status_code=201)
async def create_session():
    session = store.create()
    logger.info("Session created: %s", session.session_id)
    return SessionCreateResponse(session_id=session.session_id)


@app.delete("/voice/session/{session_id}", status_code=204)
async def delete_session(session_id: str):
    deleted = store.delete(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    logger.info("Session deleted: %s", session_id)


@app.get("/voice/session/{session_id}", response_model=SessionStateResponse)
async def get_session(session_id: str):
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    d = session.to_dict()
    return SessionStateResponse(
        session_id=session_id,
        intent=d["intent"],
        text_brain=d["text_brain"],
    )


# ------------------------------------------------------------------ #
# Audio query — session-aware (also keeps the old /voice/query path)  #
# ------------------------------------------------------------------ #

async def _run_query(service: VoiceService, req: VoiceQueryRequest, session_id: str | None):
    try:
        return await service.query_audio_base64(
            audio_base64=req.audio_base64,
            session_id=session_id,
            mode=req.mcp_mode,
            role_id=req.role_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/voice/query/{session_id}", response_model=VoiceQueryResponse)
async def voice_query_session(session_id: str, req: VoiceQueryRequest):
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    service: VoiceService = app.state.voice_service
    res = await _run_query(service, req, session_id)
    return VoiceQueryResponse(
        transcript=res.transcript,
        reply_text=res.reply_text,
        reply_audio_base64=res.reply_audio_base64,
        session_id=res.session_id,
    )


@app.post("/voice/query", response_model=VoiceQueryResponse)
async def voice_query(req: VoiceQueryRequest):
    """Sessionless fallback — backwards-compatible with existing clients."""
    service: VoiceService = app.state.voice_service
    res = await _run_query(service, req, None)
    return VoiceQueryResponse(
        transcript=res.transcript,
        reply_text=res.reply_text,
        reply_audio_base64=res.reply_audio_base64,
        session_id=res.session_id,
    )


# ------------------------------------------------------------------ #
# Push — text brain → audio queue                                     #
# ------------------------------------------------------------------ #

@app.post("/voice/push/{session_id}", response_model=PushResponse)
async def push_to_session(session_id: str, req: PushRequest):
    if req.type not in ("progress", "result", "question"):
        raise HTTPException(status_code=422, detail="type must be progress|result|question")
    service: VoiceService = app.state.voice_service
    try:
        queued = await service.push_to_session(
            session_id=session_id,
            summary=req.summary,
            push_type=req.type,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if not queued:
        raise HTTPException(status_code=404, detail="Session not found")
    return PushResponse(queued=True)


# ------------------------------------------------------------------ #
# Poll — pending audio for ChatMCP to play                            #
# ------------------------------------------------------------------ #

@app.get("/voice/pending/{session_id}", response_model=PendingAudioResponse)
async def pending_audio(session_id: str):
    service: VoiceService = app.state.voice_service
    item = service.pop_pending_audio(session_id)
    if item is None:
        return PendingAudioResponse(pending=False)
    return PendingAudioResponse(
        pending=True,
        type=item.type,
        reply_text=item.summary,
        reply_audio_base64=item.reply_audio_base64,
    )


# ------------------------------------------------------------------ #
# Poll — context updates for ChatMCP to feed back to text brain       #
# ------------------------------------------------------------------ #

@app.get("/voice/context/{session_id}", response_model=PendingContextResponse)
async def pending_context(session_id: str):
    service: VoiceService = app.state.voice_service
    item = service.pop_pending_context(session_id)
    if item is None:
        return PendingContextResponse(update=False)
    session = store.get(session_id)
    return PendingContextResponse(
        update=True,
        type=item.type,
        content=item.content,
        context_version=session.context_version if session else None,
    )
