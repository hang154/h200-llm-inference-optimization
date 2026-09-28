#!/usr/bin/env bash
set -euo pipefail
root=${EXPERIMENT_ROOT:-/home/work/timecore-ai/h200-inference-optimization-20260929}
exec >> "$root/logs/pd_smoke.log" 2>&1
mkdir -p "$root/artifacts/vllm_official" "$root/results/raw/pd_disaggregated"
export XDG_CACHE_HOME="$root/.cache/xdg"
export VLLM_CACHE_ROOT="$root/.cache/vllm"
export TORCHINDUCTOR_CACHE_DIR="$root/.cache/torchinductor"
mkdir -p "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR"

proxy="$root/artifacts/vllm_official/toy_proxy_server.py"
curl -fL --retry 5 --retry-delay 5 \
  https://raw.githubusercontent.com/vllm-project/vllm/main/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py \
  -o "$proxy"
sha256sum "$proxy" > "$proxy.sha256"

cleanup() {
  for pid in "${proxy_pid:-}" "${prefill_pid:-}" "${decode_pid:-}"; do
    [ -n "$pid" ] && kill -TERM "$pid" 2>/dev/null || true
  done
  sleep 5
}
trap cleanup EXIT INT TERM

env CUDA_VISIBLE_DEVICES=2 VLLM_NIXL_SIDE_CHANNEL_PORT=5559 UCX_NET_DEVICES=all \
  "$root/vllm_venv/bin/vllm" serve Qwen/Qwen3-8B --served-model-name Qwen/Qwen3-8B \
  --host 127.0.0.1 --port 18901 --block-size 128 --gpu-memory-utilization 0.82 \
  --max-model-len 40000 --enforce-eager \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}' &
prefill_pid=$!
env CUDA_VISIBLE_DEVICES=3 VLLM_NIXL_SIDE_CHANNEL_PORT=5659 UCX_NET_DEVICES=all \
  "$root/vllm_venv/bin/vllm" serve Qwen/Qwen3-8B --served-model-name Qwen/Qwen3-8B \
  --host 127.0.0.1 --port 18902 --block-size 128 --gpu-memory-utilization 0.82 \
  --max-model-len 40000 --enforce-eager \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}' &
decode_pid=$!

deadline=$(( $(date +%s) + 900 ))
for port in 18901 18902; do
  until curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; do
    [ "$(date +%s)" -gt "$deadline" ] && exit 1
    kill -0 "$prefill_pid" "$decode_pid"
    sleep 5
  done
done

"$root/vllm_venv/bin/python" "$proxy" --port 18900 \
  --prefiller-hosts localhost --prefiller-ports 18901 \
  --decoder-hosts localhost --decoder-ports 18902 &
proxy_pid=$!
sleep 5
kill -0 "$proxy_pid"
"$root/vllm_venv/bin/python" "$root/scripts/benchmark_openai.py" \
  --endpoint http://127.0.0.1:18900 --label pd_disaggregated \
  --context 8192 --qps 1 --requests 8 --max-tokens 128 \
  --output "$root/results/raw/pd_disaggregated/context_8192_qps_1.json"
touch "$root/results/raw/pd_disaggregated/COMPLETE"
