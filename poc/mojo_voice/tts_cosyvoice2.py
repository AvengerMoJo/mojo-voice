from __future__ import annotations

import io
import logging
import os
from pathlib import Path
import struct
import sys
import wave
from typing import Any

logger = logging.getLogger(__name__)


class CosyVoiceTTS:
    """
    CosyVoice TTS wrapper.

    Prefers CosyVoice3 runtime and falls back to CosyVoice2/CosyVoice.
    """

    def __init__(self, model_path: str, speaker: str = "中文女") -> None:
        self.model_path = _resolve_local_model_path(model_path)
        self.speaker = speaker
        self.sample_rate = 24000
        self._engine = None
        self._runtime_name = "CosyVoice"
        self._effective_speaker = speaker
        # CosyVoice3 zero-shot: path to reference WAV and instruct prefix
        self._prompt_wav: str | None = os.getenv("COSYVOICE_PROMPT_WAV") or _find_default_prompt_wav()
        self._instruct_text: str = os.getenv(
            "COSYVOICE_INSTRUCT_TEXT",
            "You are a helpful assistant. 请用普通话说。<|endofprompt|>",
        )

    def load(self) -> None:
        if self._engine is not None:
            return

        _ensure_local_cosyvoice_paths()

        try:
            from cosyvoice.cli import cosyvoice as cv_mod
        except Exception as exc:
            raise RuntimeError(
                "CosyVoice runtime not installed. Follow README setup for CosyVoice3."
            ) from exc

        cls = None
        for name in ("CosyVoice3", "CosyVoice2", "CosyVoice"):
            c = getattr(cv_mod, name, None)
            if c is not None:
                cls = c
                self._runtime_name = name
                break
        if cls is None:
            raise RuntimeError("No CosyVoice runtime class found (CosyVoice3/2/1).")

        logger.info("Loading %s model from %s", self._runtime_name, self.model_path)
        self._engine = cls(self.model_path)
        self._effective_speaker = self._resolve_speaker(self._engine, self.speaker)
        logger.info("%s speaker selected: %s", self._runtime_name, self._effective_speaker)

    def synthesize_wav(self, text: str) -> bytes:
        self.load()
        if self._engine is None:
            raise RuntimeError("TTS engine failed to load")

        chunks = []
        try:
            # CosyVoice3 has no SFT speakers — use instruct2 (zero-shot) mode
            if self._runtime_name == "CosyVoice3" or not self._effective_speaker:
                if not self._prompt_wav:
                    raise RuntimeError(
                        "CosyVoice3 requires a reference WAV. "
                        "Set COSYVOICE_PROMPT_WAV to a .wav file path."
                    )
                logger.info("Using instruct2 mode with prompt_wav=%s", self._prompt_wav)
                gen = self._engine.inference_instruct2(
                    text, self._instruct_text, self._prompt_wav, stream=False
                )
            else:
                gen = self._engine.inference_sft(text, self._effective_speaker, stream=False)

            for out in gen:
                if "tts_speech" in out:
                    chunks.append(out["tts_speech"])
            if not chunks:
                raise RuntimeError(f"{self._runtime_name} produced no audio chunks")

            try:
                import torch
                waveform = torch.cat(chunks, dim=-1).squeeze().detach().cpu().numpy()
            except Exception as exc:
                raise RuntimeError(f"Failed to materialize {self._runtime_name} output waveform") from exc

            return _float_pcm_to_wav_bytes(waveform, self.sample_rate)
        except Exception as exc:
            raise RuntimeError(f"{self._runtime_name} synthesis failed: {exc}") from exc

    @staticmethod
    def _resolve_speaker(engine: Any, preferred: str) -> str:
        if not hasattr(engine, "list_available_spks"):
            return preferred
        try:
            spks = list(engine.list_available_spks() or [])
            if not spks:
                return preferred
            if preferred in spks:
                return preferred
            return spks[0]
        except Exception:
            return preferred


def _float_pcm_to_wav_bytes(samples, sample_rate: int) -> bytes:
    # Convert float [-1, 1] array-like to 16-bit PCM WAV.
    pcm = bytearray()
    for v in samples:
        x = float(v)
        if x > 1.0:
            x = 1.0
        if x < -1.0:
            x = -1.0
        pcm.extend(struct.pack("<h", int(x * 32767)))

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(bytes(pcm))
    return buf.getvalue()


# Backward-compat alias used by older imports.
CosyVoice2TTS = CosyVoiceTTS


def _find_default_prompt_wav() -> str | None:
    this_file = os.path.abspath(__file__)
    poc_dir = os.path.dirname(os.path.dirname(this_file))
    candidate = os.path.join(poc_dir, ".vendor", "CosyVoice", "asset", "zero_shot_prompt.wav")
    if os.path.isfile(candidate):
        return candidate
    return None


def _ensure_local_cosyvoice_paths() -> None:
    """
    Ensure local vendor paths are importable without editable installs.
    This avoids py312 build issues from third_party/Matcha-TTS.
    """
    this_file = os.path.abspath(__file__)
    poc_dir = os.path.dirname(os.path.dirname(this_file))
    vendor_cosy = os.path.join(poc_dir, ".vendor", "CosyVoice")
    vendor_matcha = os.path.join(vendor_cosy, "third_party", "Matcha-TTS")
    for p in (vendor_cosy, vendor_matcha):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)


def _resolve_local_model_path(raw_path: str) -> str:
    # Never let CosyVoice treat a non-existent local-looking path as remote model id.
    configured = os.path.abspath(os.path.expanduser((raw_path or "").strip()))
    ok, missing = _check_cosyvoice_model_dir(configured)
    if ok:
        return configured

    # Try common ModelScope cache locations that already contain downloaded weights.
    cache_candidates = [
        Path(os.path.expanduser("~/.cache/modelscope/hub/FunAudioLLM/Fun-CosyVoice3-0.5B-2512")),
        Path(os.path.expanduser("~/.cache/modelscope/hub/FunAudioLLM/Fun-CosyVoice3-0___5B-2512")),
    ]
    for c in cache_candidates:
        ok_c, _missing_c = _check_cosyvoice_model_dir(str(c))
        if ok_c:
            logger.warning(
                "Configured COSYVOICE_MODEL_PATH is invalid/incomplete: %s. Using cached model: %s",
                configured,
                c,
            )
            return str(c)

    raise RuntimeError(
        "CosyVoice model path is invalid or incomplete. "
        f"Set COSYVOICE_MODEL_PATH to a fully downloaded local folder. got={configured}. "
        f"missing={missing}"
    )


def _check_cosyvoice_model_dir(model_dir: str) -> tuple[bool, list[str]]:
    p = Path(model_dir)
    if not p.is_dir():
        return False, ["<directory not found>"]
    must_exist = [
        p / "cosyvoice3.yaml",
        p / "flow.pt",
        p / "llm.pt",
        p / "speech_tokenizer_v3.onnx",
    ]
    missing = [x.name for x in must_exist if not x.exists()]
    return len(missing) == 0, missing
