#!/usr/bin/env bash
set -euo pipefail
root=${EXPERIMENT_ROOT:-/home/work/timecore-ai/h200-inference-optimization-20260929}
cd "$root/speculators"
export CUDA_VISIBLE_DEVICES=3
export HF_HOME="$root/.cache/huggingface"
export UV_CACHE_DIR="$root/.cache/uv"
export OMP_NUM_THREADS=24
exec taskset -c 24-47 "$root/speculators_venv/bin/torchrun" \
  --standalone --nproc_per_node 1 -m speculators.train \
  --verifier-name-or-path Qwen/Qwen3-8B \
  --data-path "$root/data/gsm8k_prepared" \
  --vllm-endpoint http://127.0.0.1:18800/v1 \
  --save-path "$root/artifacts/custom_eagle3" \
  --draft-vocab-size 32000 --epochs 1 --lr 1e-4 \
  --total-seq-len 2048 --on-missing generate --on-generate delete
