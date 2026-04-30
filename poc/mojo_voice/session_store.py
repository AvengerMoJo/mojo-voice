from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ContextUpdate:
    id: str
    type: str  # "clarification" | "refinement"
    content: str
    consumed: bool = False


@dataclass
class AudioQueueItem:
    id: str
    type: str  # "progress" | "result" | "question"
    summary: str
    reply_audio_base64: str
    played: bool = False


@dataclass
class MissingContext:
    field: str
    asked: bool = False
    answered: str | None = None


@dataclass
class VoiceSession:
    session_id: str
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    # intent tracking
    original_ask: str = ""
    refined_ask: str = ""
    clarity_score: float = 1.0
    missing_context: list[MissingContext] = field(default_factory=list)

    # text brain state
    text_brain_status: str = "idle"  # "idle" | "running" | "complete"
    text_brain_working_on: str = ""
    context_version: int = 0

    # queues
    audio_queue: list[AudioQueueItem] = field(default_factory=list)
    context_queue: list[ContextUpdate] = field(default_factory=list)

    def touch(self) -> None:
        self.last_active = time.time()

    def next_pending_audio(self) -> AudioQueueItem | None:
        for item in self.audio_queue:
            if not item.played:
                return item
        return None

    def mark_audio_played(self, item_id: str) -> None:
        for item in self.audio_queue:
            if item.id == item_id:
                item.played = True

    def next_pending_context(self) -> ContextUpdate | None:
        for item in self.context_queue:
            if not item.consumed:
                return item
        return None

    def mark_context_consumed(self, item_id: str) -> None:
        for item in self.context_queue:
            if item.id == item_id:
                item.consumed = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "last_active": self.last_active,
            "intent": {
                "original_ask": self.original_ask,
                "refined_ask": self.refined_ask,
                "clarity_score": self.clarity_score,
                "missing_context": [
                    {"field": m.field, "asked": m.asked, "answered": m.answered}
                    for m in self.missing_context
                ],
            },
            "text_brain": {
                "status": self.text_brain_status,
                "working_on": self.text_brain_working_on,
                "context_version": self.context_version,
            },
        }


class SessionStore:
    """Thread-safe in-memory session store with TTL eviction."""

    SESSION_TTL_SECONDS = 3600  # 1 hour

    def __init__(self) -> None:
        self._sessions: dict[str, VoiceSession] = {}
        self._lock = threading.Lock()

    def create(self) -> VoiceSession:
        session = VoiceSession(session_id=str(uuid.uuid4()))
        with self._lock:
            self._sessions[session.session_id] = session
            self._evict_expired()
        return session

    def get(self, session_id: str) -> VoiceSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def _evict_expired(self) -> None:
        cutoff = time.time() - self.SESSION_TTL_SECONDS
        expired = [sid for sid, s in self._sessions.items() if s.last_active < cutoff]
        for sid in expired:
            del self._sessions[sid]


# Global singleton — one store for the whole process
store = SessionStore()
