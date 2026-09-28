#!/usr/bin/env bash
set -euo pipefail

root=${EXPERIMENT_ROOT:-/home/work/timecore-ai/h200-inference-optimization-20260929}
label=$1
model=$2
draft=${3:-}
k=${4:-0}
kv_dtype=${5:-auto}
port=${BENCH_PORT:-18810}
server_name="serve_${label}"
mkdir -p "$root/results/raw/$label" "$root/logs"
export XDG_CACHE_HOME="$root/.cache/xdg"
export VLLM_CACHE_ROOT="$root/.cache/vllm"
export TORCHINDUCTOR_CACHE_DIR="$root/.cache/torchinductor"
mkdir -p "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR"

args=("$root/vllm_venv/bin/vllm" serve "$model" --served-model-name Qwen/Qwen3-8B --host 127.0.0.1 --port "$port"
  --gpu-memory-utilization 0.82 --max-model-len 40000 --max-num-seqs 32
  --trust-remote-code --generation-config vllm)
if [ "$kv_dtype" != auto ]; then args+=(--kv-cache-dtype "$kv_dtype"); fi
if [ -n "$draft" ]; then
  args+=(--speculative-config "{\"method\":\"eagle3\",\"model\":\"$draft\",\"num_speculative_tokens\":$k}")
fi

nohup bash "$root/scripts/remote_guardian.sh" "$server_name" \
  taskset -c 0-23 env CUDA_VISIBLE_DEVICES=2 HF_HOME="$root/.cache/huggingface" XDG_CACHE_HOME="$XDG_CACHE_HOME" VLLM_CACHE_ROOT="$VLLM_CACHE_ROOT" TORCHINDUCTOR_CACHE_DIR="$TORCHINDUCTOR_CACHE_DIR" \
  "${args[@]}" > "$root/logs/${server_name}_launcher.log" 2>&1 < /dev/null &
guardian=$!
echo "$guardian" > "$root/runtime/${server_name}.guardian.pid"

cleanup() {
  child=$(cat "$root/runtime/${server_name}.child.pid" 2>/dev/null || true)
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

curl -fsS "http://127.0.0.1:$port/metrics" > "$root/results/raw/$label/metrics_before.txt" || true
for context in 512 8192 32768; do
  for qps in 1 8; do
    [ -f "$root/runtime/STOP_REQUESTED" ] && exit 75
    requests=12
    [ "$context" -eq 32768 ] && requests=8
    "$root/vllm_venv/bin/python" "$root/scripts/benchmark_openai.py" \
      --endpoint "http://127.0.0.1:$port" --label "$label" --context "$context" \
      --qps "$qps" --requests "$requests" --max-tokens 128 \
      --output "$root/results/raw/$label/context_${context}_qps_${qps}.json"
  done
done
curl -fsS "http://127.0.0.1:$port/metrics" > "$root/results/raw/$label/metrics_after.txt" || true
touch "$root/results/raw/$label/COMPLETE"
