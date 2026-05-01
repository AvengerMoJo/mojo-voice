# mojo-voice

Quick, demoable voice PoC that bridges a mobile-ready speech runtime (ExecuTorch adapter) to MoJoAssistant MCP over HTTP JSON-RPC.

## Scope

This repo is intentionally small:
- Keep your existing MCP server in `MoJoAssistant`.
- Provide a direct client bridge from speech pipeline -> MCP `tools/call`.
- Keep runtime pluggable so you can swap mock runtime with real ExecuTorch mobile runtime.

## PoC Flow

1. STT captures user utterance (`transcribe_from_mic`)  
2. PoC calls MoJo MCP `tools/call` (mode-driven):
   - default `search_memory` mode (supported MCP path)
   - optional `dialog` mode (legacy; may be disabled in MCP depending on server policy)
3. Parse role response text
4. TTS reads assistant output (`speak`)

## Layout

- `poc/run_poc.py` - runnable demo loop
- `poc/run_voice_api.py` - HTTP API server for frontend (`audio_base64` in/out)
- `poc/mojo_voice/mcp_client.py` - MCP JSON-RPC client
- `poc/mojo_voice/runtime.py` - runtime interface contracts
- `poc/mojo_voice/runtime_mock.py` - keyboard + console TTS mock runtime
- `poc/mojo_voice/runtime_executorch.py` - ExecuTorch-ready adapter skeleton
- `poc/mojo_voice/pipeline.py` - orchestration loop
- `poc/mojo_voice/stt_funasr.py` - in-repo FunASR STT implementation
- `poc/mojo_voice/funasr_transcription.py` - FunASR tag cleanup helper
- `poc/mojo_voice/tts_cosyvoice2.py` - CosyVoice TTS implementation (CosyVoice3-first)
- `poc/mojo_voice/voice_api.py` - FastAPI endpoints
- `poc/scripts/install_cosyvoice_rocm.sh` - CosyVoice installer that preserves active ROCm torch

## Quick Demo

```bash
cd /home/alex/Development/Personal/MoJoAssistant
source venv/bin/activate
cd submodules/mojo-voice/poc
pip install -r requirements.txt
cp .env.example .env
python run_poc.py
```

Required env vars:
- `MOJO_MCP_URL` (default `http://127.0.0.1:8000/`)
- `MOJO_ROLE_ID` (default `assistant`)
- `MOJO_MCP_API_KEY` (optional; falls back to `MCP_API_KEY` if present)

Optional:
- `MOJO_VOICE_RUNTIME=mock|executorch` (default `mock`)
- `MOJO_MCP_MODE=search_memory|dialog` (default `search_memory`)
- `MOJO_STT_PROVIDER=funasr|zai_asr` (default `funasr`, strict fail if misconfigured)

## Audio Base64 API (Frontend -> Backend)

This mode is for frontend apps that send recorded audio as base64 JSON.

### Start API server

```bash
cd /home/alex/Development/Personal/MoJoAssistant
source venv/bin/activate
cd submodules/mojo-voice/poc
pip install -r requirements.txt
cp .env.example .env
python run_voice_api.py
```

### Endpoint

- `POST /voice/query`
- Body:
```json
{
  "audio_base64": "<base64-audio-or-data-uri>",
  "mcp_mode": "search_memory",
  "role_id": "assistant"
}
```

- `POST /voice/transcribe` (STT only, no TTS/MCP side effects)
- Body:
```json
{
  "audio_base64": "<base64-audio-or-data-uri>"
}
```

### Response

```json
{
  "transcript": "...",
  "reply_text": "...",
  "reply_audio_base64": "...",
  "reply_audio_format": "wav",
  "session_id": null
}
```

## Setup FunASR + CosyVoice3 (ROCm-safe)

Use the same MoJoAssistant venv so CosyVoice shares the existing ROCm torch stack:

```bash
cd /home/alex/Development/Personal/MoJoAssistant
source venv/bin/activate
cd submodules/mojo-voice/poc
pip install -r requirements.txt
./scripts/install_cosyvoice_rocm.sh
```

The installer avoids reinstalling `torch/torchaudio/torchvision` and keeps the active ROCm build.
It also validates `CosyVoice3` import.

Optional one-shot model download via installer:

```bash
COSYVOICE_DOWNLOAD_MODEL=1 \
COSYVOICE_DOWNLOAD_SOURCE=modelscope \
COSYVOICE_MODEL_ID=FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
COSYVOICE_MODEL_DIR=/path/to/pretrained_models/Fun-CosyVoice3-0.5B \
./scripts/install_cosyvoice_rocm.sh
```

After install, set:

- `COSYVOICE_MODEL_PATH` in `.env`
- `COSYVOICE_SPEAKER` in `.env`

If these are not installed/configured, `/voice/query` will return a setup error.

## Z.AI GLM-ASR provider path

Use cloud ASR without changing your voice API contract:

```bash
export MOJO_STT_PROVIDER=zai_asr
export ZAI_API_KEY=...
export ZAI_ASR_MODEL=glm-asr-2512
```

The service fails fast on invalid provider config and does not silently fall back.

## Mobile Integration Notes

`runtime_executorch.py` is the seam where you bind real mobile audio I/O + ExecuTorch model calls:
- iOS: Swift/ObjC audio capture/playback + C++ bridge
- Android: AudioRecord/AudioTrack + JNI bridge
- Replace mock STT/TTS methods with on-device model inference calls

The MCP client can be reused as-is in a native app (same JSON-RPC payload).

## Status Snapshot (2026-04-29)

Current implementation status for this PoC:

- `POST /voice/query` is implemented with JSON `audio_base64` input/output.
- STT path is local FunASR in this repo (`poc/mojo_voice/stt_funasr.py`).
- TTS path is CosyVoice wrapper (`poc/mojo_voice/tts_cosyvoice2.py`) with CosyVoice3-first loading.
- MCP integration uses `tools/call` with default `search_memory` mode.
- `dialog` mode is optional and may be blocked depending on MCP server policy.

Recent reliability fixes already applied:

- Added `scripts/prepare_cosyvoice3_model.sh` to resume/download CosyVoice3 weights to a fixed local folder and validate required files.
- Added strict local model-path validation so invalid/incomplete paths fail fast instead of silently re-triggering remote downloads.
- Added cache fallback logic for common ModelScope local paths.
- Changed default speaker handling to auto-select first available speaker if preferred one is not valid.

Known blocker before stable speech-to-speech testing:

- CosyVoice3 model directory must be fully downloaded (required files include `flow.pt`, `llm.pt`, `speech_tokenizer_v3.onnx`, `cosyvoice3.yaml`).
- If model is incomplete, API startup will fail with an explicit missing-files error.

Recommended resume sequence:

1. Activate MoJoAssistant venv.
2. Run `./scripts/prepare_cosyvoice3_model.sh` under `submodules/mojo-voice/poc`.
3. Set `.env`:
   - `COSYVOICE_MODEL_PATH=<validated local model dir>`
   - `COSYVOICE_SPEAKER=`
4. Start API with `python run_voice_api.py`.
5. Test from client with `curl` to `/voice/query`.
