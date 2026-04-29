#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="${ROOT_DIR}/.vendor/CosyVoice"
REQ_FILTERED="/tmp/cosyvoice_req_filtered.txt"
MODEL_ID="${COSYVOICE_MODEL_ID:-FunAudioLLM/Fun-CosyVoice3-0.5B-2512}"
MODEL_DIR="${COSYVOICE_MODEL_DIR:-${ROOT_DIR}/pretrained_models/Fun-CosyVoice3-0.5B}"
DOWNLOAD_MODEL="${COSYVOICE_DOWNLOAD_MODEL:-0}"      # 1 to enable auto-download
DOWNLOAD_SOURCE="${COSYVOICE_DOWNLOAD_SOURCE:-modelscope}"  # modelscope|huggingface

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "error: activate MoJoAssistant venv first (source /home/alex/Development/Personal/MoJoAssistant/venv/bin/activate)"
  exit 1
fi

echo "[1/7] verify python/torch runtime"
python - <<'PY'
import sys
try:
    import torch
except Exception as e:
    raise SystemExit(f"error: torch not available in active venv: {e}")
hip = getattr(torch.version, 'hip', None)
print(f"python={sys.version.split()[0]} torch={torch.__version__} hip={hip}")
if not hip:
    raise SystemExit("error: active torch is not ROCm-enabled (torch.version.hip is empty)")
PY

echo "[2/7] ensure CosyVoice source present"
mkdir -p "${ROOT_DIR}/.vendor"
if [[ ! -d "${VENDOR_DIR}/.git" ]]; then
  git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git "${VENDOR_DIR}"
else
  git -C "${VENDOR_DIR}" pull --ff-only || true
  git -C "${VENDOR_DIR}" submodule update --init --recursive
fi

echo "[3/7] bootstrap packaging tools"
python -m pip install -U pip setuptools wheel
python -m pip install "setuptools<81"

echo "[4/7] install CosyVoice deps without conflicting torch pins"
# Remove pinned packages known to conflict with ROCm torch / local platform.
# Also skip grpcio-tools/openai-whisper/protobuf pins to avoid cross-stack breakage.
grep -Ev '^(torch|torchaudio|torchvision|grpcio|grpcio-tools|openai-whisper|protobuf)==' "${VENDOR_DIR}/requirements.txt" > "${REQ_FILTERED}"
python -m pip install -r "${REQ_FILTERED}"

echo "[5/7] install compatible grpcio stack (ROCm-safe, no triton override)"
python -m pip install "grpcio>=1.62,<2"
python -m pip install "grpcio-tools>=1.62,<2"

echo "[6/7] register CosyVoice source path in current venv"
python - <<PY
import site
from pathlib import Path

vendor = Path("${VENDOR_DIR}").resolve()
pth_dir = Path(site.getsitepackages()[0])
pth_file = pth_dir / "cosyvoice_local.pth"
pth_file.write_text(str(vendor) + "\\n", encoding="utf-8")
print(f"wrote {pth_file} -> {vendor}")
PY

python - <<'PY'
from cosyvoice.cli.cosyvoice import CosyVoice3
print('CosyVoice3 import OK')
PY

echo "[7/7] optional CosyVoice3 model download"
if [[ "${DOWNLOAD_MODEL}" == "1" ]]; then
  MODEL_ID="${MODEL_ID}" MODEL_DIR="${MODEL_DIR}" DOWNLOAD_SOURCE="${DOWNLOAD_SOURCE}" python - <<PY
import os
from pathlib import Path

model_id = os.environ.get("MODEL_ID")
model_dir = os.environ.get("MODEL_DIR")
source = os.environ.get("DOWNLOAD_SOURCE", "modelscope").strip().lower()
dst = Path(model_dir)
dst.mkdir(parents=True, exist_ok=True)

if source == "huggingface":
    from huggingface_hub import snapshot_download
    snapshot_download(model_id, local_dir=str(dst))
elif source == "modelscope":
    from modelscope import snapshot_download
    snapshot_download(model_id, local_dir=str(dst))
else:
    raise SystemExit(f"Unsupported COSYVOICE_DOWNLOAD_SOURCE={source}")
print(f"model ready: {dst}")
PY
else
  echo "skip model download (set COSYVOICE_DOWNLOAD_MODEL=1 to enable)"
fi

echo "done: CosyVoice installed into active venv (${VIRTUAL_ENV})"
echo "set COSYVOICE_MODEL_PATH in .env to your model directory before running run_voice_api.py"
echo "suggested model path: ${MODEL_DIR}"
