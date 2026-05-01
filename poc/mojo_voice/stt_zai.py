from __future__ import annotations

import base64
import json
import logging
import os
import random
import time
import uuid

import httpx

from .stt_base import STTResult

logger = logging.getLogger(__name__)


class ZaiASRSTT:
    """
    Z.AI GLM-ASR provider-backed STT engine.

    Endpoint defaults to:
    https://api.z.ai/api/paas/v4/audio/transcriptions
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "glm-asr-2512",
        endpoint: str = "https://api.z.ai/api/paas/v4/audio/transcriptions",
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        retry_base_seconds: float = 0.6,
        retry_max_seconds: float = 5.0,
        user_id: str | None = None,
        hotwords: list[str] | None = None,
    ) -> None:
        self.api_key = api_key.strip()
        self.model = model.strip() or "glm-asr-2512"
        self.endpoint = endpoint.strip()
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(0, max_retries)
        self.retry_base_seconds = max(0.05, retry_base_seconds)
        self.retry_max_seconds = max(self.retry_base_seconds, retry_max_seconds)
        self.user_id = user_id.strip() if user_id else None
        self.hotwords = hotwords or []

    @classmethod
    def from_env(cls) -> "ZaiASRSTT":
        api_key = os.getenv("ZAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("ZAI_API_KEY is required when MOJO_STT_PROVIDER=zai_asr")
        endpoint = os.getenv(
            "ZAI_ASR_ENDPOINT",
            "https://api.z.ai/api/paas/v4/audio/transcriptions",
        ).strip()
        model = os.getenv("ZAI_ASR_MODEL", "glm-asr-2512").strip()
        timeout_seconds = float(os.getenv("ZAI_ASR_TIMEOUT_SECONDS", "30").strip())
        max_retries = int(os.getenv("ZAI_ASR_MAX_RETRIES", "3").strip())
        retry_base_seconds = float(
            os.getenv("ZAI_ASR_RETRY_BASE_SECONDS", "0.6").strip()
        )
        retry_max_seconds = float(
            os.getenv("ZAI_ASR_RETRY_MAX_SECONDS", "5.0").strip()
        )
        user_id = os.getenv("ZAI_ASR_USER_ID", "").strip() or None
        hotwords_raw = os.getenv("ZAI_ASR_HOTWORDS", "").strip()
        hotwords = [w.strip() for w in hotwords_raw.split(",") if w.strip()]
        return cls(
            api_key=api_key,
            model=model,
            endpoint=endpoint,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_base_seconds=retry_base_seconds,
            retry_max_seconds=retry_max_seconds,
            user_id=user_id,
            hotwords=hotwords,
        )

    def load(self) -> None:
        # Cloud provider has no heavy local preload, but we validate config early.
        if not self.endpoint:
            raise RuntimeError("ZAI_ASR_ENDPOINT must not be empty")
        if not self.api_key:
            raise RuntimeError("ZAI_API_KEY must not be empty")

    def transcribe_wav_path(self, wav_path: str) -> STTResult:
        self.load()
        headers = {"Authorization": f"Bearer {self.api_key}"}
        with open(wav_path, "rb") as fh:
            raw = fh.read()
        req_payload: dict[str, object] = {
            "model": self.model,
            "stream": False,
            "file_base64": base64.b64encode(raw).decode("utf-8"),
            "request_id": str(uuid.uuid4()),
        }
        if self.user_id:
            req_payload["user_id"] = self.user_id
        if self.hotwords:
            req_payload["hotwords"] = self.hotwords

        resp: httpx.Response | None = None
        last_req_id = str(req_payload["request_id"])
        for attempt in range(self.max_retries + 1):
            resp = self._post_json(headers=headers, payload=req_payload)
            log_id = resp.headers.get("X-LOG-ID", "")
            if log_id:
                last_req_id = log_id
            if resp.status_code == 429:
                if attempt >= self.max_retries:
                    break
                sleep_s = min(
                    self.retry_max_seconds,
                    self.retry_base_seconds * (2**attempt) + random.uniform(0, 0.35),
                )
                logger.warning(
                    "Z.AI ASR 429 rate_limited; retry attempt=%s/%s sleep=%.2fs request_id=%s",
                    attempt + 1,
                    self.max_retries + 1,
                    sleep_s,
                    last_req_id,
                )
                time.sleep(sleep_s)
                continue
            break
        assert resp is not None
        if resp.status_code == 429:
            raise RuntimeError(
                f"Z.AI ASR rate_limited after retries; request_id={last_req_id}"
            )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Z.AI ASR request failed: status={resp.status_code} request_id={last_req_id} body={resp.text[:300]}"
            )

        try:
            payload = resp.json()
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Z.AI ASR returned invalid JSON: {resp.text[:200]}") from exc

        text = self._extract_text(payload).strip()
        if not text:
            raise RuntimeError("Z.AI ASR returned empty transcript")
        logger.info("Z.AI ASR transcript length=%s", len(text))
        return STTResult(text=text, raw=payload)

    def _post_json(self, *, headers: dict[str, str], payload: dict) -> httpx.Response:
        req_headers = dict(headers)
        req_headers["Content-Type"] = "application/json"
        with httpx.Client(timeout=self.timeout_seconds) as client:
            return client.post(self.endpoint, headers=req_headers, json=payload)

    @staticmethod
    def _extract_text(payload: object) -> str:
        if isinstance(payload, dict):
            for k in ("text", "transcript", "result"):
                v = payload.get(k)
                if isinstance(v, str) and v.strip():
                    return v
            data = payload.get("data")
            if isinstance(data, dict):
                for k in ("text", "transcript", "result"):
                    v = data.get(k)
                    if isinstance(v, str) and v.strip():
                        return v
        return ""
