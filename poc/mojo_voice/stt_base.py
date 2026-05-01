from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class STTResult:
    text: str
    raw: object


class STTEngine(Protocol):
    def load(self) -> None: ...

    def transcribe_wav_path(self, wav_path: str) -> STTResult: ...
