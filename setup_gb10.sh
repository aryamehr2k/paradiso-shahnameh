#!/usr/bin/env bash
# ============================================================
# Persian-miniature (Shahnameh) Flux LoRA  ·  env setup for DGX Spark (GB10 / Blackwell)
# ------------------------------------------------------------
# GB10 is Blackwell = sm_121. As of late 2025, stock PyTorch 2.8 / NGC 25.09
# could NOT compile kernels for Flux training on this arch. The CES-2026 DGX
# software release fixed most of it. Two paths below — try A, fall back to B.
# ============================================================
set -e

# ---------- PATH A: cu130 nightly wheels in a clean venv (simplest) ----------
# aarch64 cu130 wheels exist for the Spark.
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# Sanity check. You WILL see a "cuda capability 12.1 / max supported 12.0"
# warning — it is safe to ignore on GB10.
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda?", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0))
PY

# Ostris ai-toolkit = cleanest Flux LoRA trainer
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
git submodule update --init --recursive
uv pip install -r requirements.txt
# re-pin cu130 torch in case requirements.txt pulled a different build:
uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
cd ..

# scraper deps
uv pip install requests pillow

# Two Blackwell gotchas, already handled in the configs:
#   1) flash-attn fails to build on GB10 AND is slower than SDPA here -> do NOT install it.
#   2) bitsandbytes 8-bit optimizers are flaky on Blackwell -> the train config uses plain adamw.

# FLUX.1-dev is gated on Hugging Face. Accept the license once, then:
#   huggingface-cli login

echo "Env ready.  dataset -> train -> generate."

# ---------- PATH B (fallback if A fights you): NVIDIA NGC container ----------
# The 25.11+ PyTorch container ships sm_121 + Transformer Engine 2.9.
#   docker pull nvcr.io/nvidia/pytorch:25.11-py3
#   docker run --gpus all -it --rm -v "$PWD":/work -w /work nvcr.io/nvidia/pytorch:25.11-py3
#   # inside:  pip install -r ai-toolkit/requirements.txt requests pillow "numpy<2"
