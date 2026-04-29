from __future__ import annotations

import base64
import re
import subprocess
import tempfile
from pathlib import Path


_DATA_URI_PREFIX = re.compile(r"^data:[^;]+;base64,")


def decode_audio_base64(audio_base64: str) -> bytes:
    payload = _DATA_URI_PREFIX.sub("", audio_base64.strip())
    try:
        return base64.b64decode(payload, validate=True)
    except Exception as exc:
        raise ValueError("Invalid audio_base64 payload") from exc


def to_wav16k_mono_bytes(audio_bytes: bytes) -> bytes:
    """Convert arbitrary audio bytes to WAV PCM16 16k mono via ffmpeg."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        "pipe:0",
        "-f",
        "wav",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        "pipe:1",
    ]
    try:
        proc = subprocess.run(
            cmd,
            input=audio_bytes,
            capture_output=True,
            check=False,
            timeout=30,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg not found. Install ffmpeg for audio conversion.") from exc

    if proc.returncode != 0 or len(proc.stdout) < 44:
        err = proc.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffmpeg conversion failed: {err[:300]}")
    return proc.stdout


def write_temp_wav(wav_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(prefix="mojo_voice_", suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        return f.name


def encode_audio_base64(audio_bytes: bytes) -> str:
    return base64.b64encode(audio_bytes).decode("utf-8")


def safe_unlink(path: str) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except Exception:
        pass
