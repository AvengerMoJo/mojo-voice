#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ID="${COSYVOICE_MODEL_ID:-FunAudioLLM/Fun-CosyVoice3-0.5B-2512}"
MODEL_DIR="${COSYVOICE_MODEL_PATH:-${COSYVOICE_MODEL_DIR:-/home/alex/.cache/modelscope/hub/FunAudioLLM/Fun-CosyVoice3-0___5B-2512}}"
DOWNLOAD_SOURCE="${COSYVOICE_DOWNLOAD_SOURCE:-modelscope}" # modelscope | huggingface

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "error: activate MoJoAssistant venv first:"
  echo "  source /home/alex/Development/Personal/MoJoAssistant/venv/bin/activate"
  exit 1
fi

echo "[1/4] checking python env"
python - <<'PY'
import sys
print(f"python={sys.version.split()[0]}")
try:
    import modelscope  # noqa: F401
except Exception as exc:
    raise SystemExit(f"error: modelscope is not installed in this venv: {exc}")
PY

echo "[2/4] downloading/resuming CosyVoice3 model"
MODEL_ID="${MODEL_ID}" MODEL_DIR="${MODEL_DIR}" DOWNLOAD_SOURCE="${DOWNLOAD_SOURCE}" python - <<'PY'
import os
from pathlib import Path

model_id = os.environ["MODEL_ID"].strip()
model_dir = Path(os.environ["MODEL_DIR"]).expanduser().resolve()
source = os.environ["DOWNLOAD_SOURCE"].strip().lower()
model_dir.mkdir(parents=True, exist_ok=True)

print(f"model_id={model_id}")
print(f"model_dir={model_dir}")
print(f"source={source}")

if source == "modelscope":
    from modelscope import snapshot_download
    snapshot_download(model_id, local_dir=str(model_dir))
elif source == "huggingface":
    from huggingface_hub import snapshot_download
    snapshot_download(model_id, local_dir=str(model_dir))
else:
    raise SystemExit(f"Unsupported COSYVOICE_DOWNLOAD_SOURCE={source}")
PY

echo "[3/4] validating required files"
MODEL_DIR="${MODEL_DIR}" python - <<'PY'
import os
from pathlib import Path

model_dir = Path(os.environ["MODEL_DIR"]).expanduser().resolve()
required = [
    "cosyvoice3.yaml",
    "flow.pt",
    "llm.pt",
    "speech_tokenizer_v3.onnx",
]
missing = [name for name in required if not (model_dir / name).exists()]
if missing:
    raise SystemExit(f"error: model incomplete. missing={missing}")
print("model validation OK")
PY

echo "[4/4] done"
echo "Set these in ${ROOT_DIR}/.env:"
echo "COSYVOICE_MODEL_PATH=${MODEL_DIR}"
echo "COSYVOICE_SPEAKER="
