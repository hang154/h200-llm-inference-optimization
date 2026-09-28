#!/usr/bin/env bash
set -euo pipefail
root=${EXPERIMENT_ROOT:-/home/work/timecore-ai/h200-inference-optimization-20260929}
label=$1
model=$2
port=18830
out="$root/results/quality/$label"
mkdir -p "$out"
[ -f "$out/COMPLETE" ] && exit 0
name="quality_${label}"
nohup bash "$root/scripts/remote_guardian.sh" "$name" \
  taskset -c 0-23 env CUDA_VISIBLE_DEVICES=2 HF_HOME="$root/.cache/huggingface" \
  XDG_CACHE_HOME="$root/.cache/xdg" VLLM_CACHE_ROOT="$root/.cache/vllm" \
  TORCHINDUCTOR_CACHE_DIR="$root/.cache/torchinductor" \
  "$root/vllm_venv/bin/vllm" serve "$model" --served-model-name Qwen/Qwen3-8B \
  --host 127.0.0.1 --port "$port" --gpu-memory-utilization 0.82 --max-model-len 40960 \
  --max-num-seqs 16 --generation-config vllm > "$root/logs/${name}_launcher.log" 2>&1 < /dev/null &
guardian=$!
printf '%s\n' "$guardian" > "$root/runtime/${name}.guardian.pid"
cleanup(){
  child=$(cat "$root/runtime/${name}.child.pid" 2>/dev/null || true)
  [ -n "$child" ] && kill -TERM -- "-$child" 2>/dev/null || true
  kill -TERM "$guardian" 2>/dev/null || true
  sleep 5
}
trap cleanup EXIT
deadline=$(( $(date +%s) + 900 ))
until curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; do
  [ -f "$root/runtime/STOP_REQUESTED" ] && exit 75
  [ "$(date +%s)" -gt "$deadline" ] && exit 1
  sleep 5
done
"$root/vllm_venv/bin/python" "$root/scripts/benchmark_quality.py" \
  --endpoint "http://127.0.0.1:$port" --label "$label" --output "$out/gsm8k_quality.json"
touch "$out/COMPLETE"
