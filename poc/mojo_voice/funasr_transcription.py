"""FunASR result parsing utilities for mojo-voice."""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class FunASRTranscription:
    """Helper for extracting clean text from FunASR/SenseVoice outputs."""

    @staticmethod
    def extract_text(raw_result: Any) -> str | None:
        if raw_result is None:
            logger.warning("FunASR result is None")
            return None

        text_to_clean: str | None = None
        if isinstance(raw_result, list) and raw_result:
            item = raw_result[0]
            if isinstance(item, dict):
                text_to_clean = str(item.get("text", ""))
        elif isinstance(raw_result, dict):
            text_to_clean = str(raw_result.get("text", ""))
        elif isinstance(raw_result, str):
            text_to_clean = raw_result

        if not text_to_clean:
            logger.warning("Could not extract text from result type=%s", type(raw_result))
            return None

        clean_text = re.sub(r"<\|[^|]+\|>", "", text_to_clean).strip()
        return clean_text or None
