from __future__ import annotations

import logging

from .funasr_transcription import FunASRTranscription
from .stt_base import STTResult

logger = logging.getLogger(__name__)


class FunASRSTT:
    """
    Local FunASR STT engine.

    Requires `funasr` package and model assets available to AutoModel.
    """

    def __init__(
        self,
        model: str = "iic/SenseVoiceSmall",
        vad_model: str | None = "fsmn-vad",
        punc_model: str | None = "ct-punc",
        device: str = "cpu",
    ) -> None:
        self.model_name = model
        self.vad_model = vad_model
        self.punc_model = punc_model
        self.device = device
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from funasr import AutoModel
        except Exception as exc:
            raise RuntimeError(
                "funasr is not installed. Install dependencies for STT first."
            ) from exc

        kwargs = {
            "model": self.model_name,
            "device": self.device,
        }
        if self.vad_model:
            kwargs["vad_model"] = self.vad_model
        if self.punc_model:
            kwargs["punc_model"] = self.punc_model

        logger.info("Loading FunASR model=%s device=%s", self.model_name, self.device)
        self._model = AutoModel(**kwargs)

    def transcribe_wav_path(self, wav_path: str) -> STTResult:
        self.load()
        assert self._model is not None
        raw = self._model.generate(input=wav_path)
        text = FunASRTranscription.extract_text(raw)
        if not text:
            text = ""
        return STTResult(text=text, raw=raw)
