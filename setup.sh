#!/usr/bin/env bash
# Creates .venv and installs Llmzempic with training support (Linux / macOS).
# Override the PyTorch build with TORCH_INDEX_URL (e.g. https://download.pytorch.org/whl/rocm6.4 for AMD).
# Set WITH_UNSLOTH=1 to also install Unsloth (faster training on NVIDIA GPUs).
set -euo pipefail
cd "$(dirname "$0")"

"${PYTHON:-python3}" -m venv .venv
PY=.venv/bin/python
"$PY" -m pip install --upgrade pip

EXTRAS="train"
DEFAULT_INDEX="https://download.pytorch.org/whl/cpu"
if command -v nvidia-smi >/dev/null 2>&1; then
	EXTRAS="train,cuda"
	DEFAULT_INDEX="https://download.pytorch.org/whl/cu128"
fi
INDEX="${TORCH_INDEX_URL:-$DEFAULT_INDEX}"
echo "Installing PyTorch from $INDEX"
"$PY" -m pip install torch --index-url "$INDEX"
"$PY" -m pip install -e ".[$EXTRAS]"
if [ "${WITH_UNSLOTH:-0}" = "1" ]; then
	"$PY" -m pip install unsloth
fi

echo
echo "Done. Start Llmzempic with ./start.sh"
