#!/usr/bin/env bash
set -euo pipefail
root=${EXPERIMENT_ROOT:-/home/work/timecore-ai/h200-inference-optimization-20260929}
label=$1 model=$2 draft=${3:-} k=${4:-0}
port=${DOMAIN_PORT:-18820} gpu=${DOMAIN_GPU:-3} cpus=${DOMAIN_CPUS:-24-47}
out="$root/results/domain/$label" name="domain_${label}"
mkdir -p "$out" "$root/logs"
export XDG_CACHE_HOME="$root/.cache/xdg" VLLM_CACHE_ROOT="$root/.cache/vllm" TORCHINDUCTOR_CACHE_DIR="$root/.cache/torchinductor"
args=("$root/vllm_venv/bin/vllm" serve "$model" --served-model-name Qwen/Qwen3-8B --host 127.0.0.1 --port "$port"
  --gpu-memory-utilization 0.82 --max-model-len 4096 --max-num-seqs 32 --generation-config vllm)
[ -n "$draft" ] && args+=(--speculative-config "{\"method\":\"eagle3\",\"model\":\"$draft\",\"num_speculative_tokens\":$k}")
nohup bash "$root/scripts/remote_guardian.sh" "$name" taskset -c "$cpus" env CUDA_VISIBLE_DEVICES="$gpu" \
  HF_HOME="$root/.cache/huggingface" XDG_CACHE_HOME="$XDG_CACHE_HOME" VLLM_CACHE_ROOT="$VLLM_CACHE_ROOT" TORCHINDUCTOR_CACHE_DIR="$TORCHINDUCTOR_CACHE_DIR" \
  "${args[@]}" > "$root/logs/${name}_launcher.log" 2>&1 < /dev/null &
guardian=$!; echo "$guardian" > "$root/runtime/${name}.guardian.pid"
cleanup(){ child=$(cat "$root/runtime/${name}.child.pid" 2>/dev/null||true); [ -n "$child" ]&&kill -TERM -- "-$child" 2>/dev/null||true; kill -TERM "$guardian" 2>/dev/null||true; sleep 4; }; trap cleanup EXIT
deadline=$(( $(date +%s)+900 )); until curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; do [ "$(date +%s)" -gt "$deadline" ]&&exit 1; sleep 5; done
curl -fsS "http://127.0.0.1:$port/metrics" > "$out/metrics_before.txt"
"$root/speculators_venv/bin/python" "$root/scripts/benchmark_gsm8k.py" --endpoint "http://127.0.0.1:$port" --label "$label" --output "$out/gsm8k_qps4.json"
curl -fsS "http://127.0.0.1:$port/metrics" > "$out/metrics_after.txt"
touch "$out/COMPLETE"
