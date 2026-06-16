#!/usr/bin/env bash
set -euo pipefail

PY="${CLEANVIDEO_FLASHVSR_WSL_PY:-$HOME/.cleanvideo/flashvsr-wsl/.venv/bin/python}"
CUDA_PREFIX="${CLEANVIDEO_FLASHVSR_WSL_CUDA:-$HOME/.cleanvideo/cuda-12.4}"

if [ ! -x "$PY" ]; then
  echo "Missing FlashVSR WSL Python: $PY" >&2
  exit 127
fi

SITE="$("$PY" -c 'import site; print(site.getsitepackages()[0])')"
NVIDIA_LIBS=""
if [ -d "$SITE/nvidia" ]; then
  NVIDIA_LIBS="$(find "$SITE/nvidia" -maxdepth 3 -type d \( -name lib -o -name lib64 \) -printf '%p:' 2>/dev/null || true)"
  NVIDIA_LIBS="${NVIDIA_LIBS%:}"
fi

export CUDA_HOME="$CUDA_PREFIX"
export PATH="$CUDA_PREFIX/bin:$CUDA_PREFIX/x86_64-conda-linux-gnu/bin:$PATH"
export LD_LIBRARY_PATH="$SITE/torch/lib:$CUDA_PREFIX/targets/x86_64-linux/lib:$CUDA_PREFIX/lib:$CUDA_PREFIX/lib64${NVIDIA_LIBS:+:$NVIDIA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false

exec "$PY" "$@"
