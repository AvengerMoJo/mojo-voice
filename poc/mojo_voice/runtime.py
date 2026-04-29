from __future__ import annotations

from typing import Protocol


class SpeechRuntime(Protocol):
    async def transcribe_from_mic(self) -> str:
        """Capture one utterance from microphone and return text."""

    async def speak(self, text: str) -> None:
        """Speak assistant output text."""
