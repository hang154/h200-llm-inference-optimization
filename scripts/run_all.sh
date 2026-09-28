#!/usr/bin/env bash
set -euo pipefail

root=${EXPERIMENT_ROOT:-/home/work/timecore-ai/h200-inference-optimization-20260929}
cd "$root"
export HF_HOME="$root/.cache/huggingface"
export UV_CACHE_DIR="$root/.cache/uv"
export TMPDIR="$root/.cache/tmp"
export XDG_CACHE_HOME="$root/.cache/xdg"
export VLLM_CACHE_ROOT="$root/.cache/vllm"
export TORCHINDUCTOR_CACHE_DIR="$root/.cache/torchinductor"
mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$TMPDIR" "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" runtime logs artifacts data results/raw telemetry

stage() {
  printf '{"stage":"%s","updated_at":"%s"}\n' "$1" "$(date --iso-8601=seconds)" > runtime/orchestrator_state.json.tmp
  mv runtime/orchestrator_state.json.tmp runtime/orchestrator_state.json
}
tree_pids() {
  local parent=$1 child
  for child in $(pgrep -P "$parent" 2>/dev/null || true); do tree_pids "$child"; done
  printf '%s\n' "$parent"
}
stop_tree() {
  local parent=$1 pids
  [ -e "/proc/$parent" ] || return 0
  pids=$(tree_pids "$parent")
  # Stop the restart-owning parent first, then its captured descendants.
  kill -KILL "$parent" 2>/dev/null || true
  for p in $pids; do [ "$p" = "$parent" ] || kill -TERM "$p" 2>/dev/null || true; done
  sleep 3
  for p in $pids; do [ "$p" = "$parent" ] || kill -KILL "$p" 2>/dev/null || true; done
}
stop_registered() {
  local name=$1 guardian child
  guardian=$(cat "runtime/${name}.guardian.pid" 2>/dev/null || true)
  child=$(cat "runtime/${name}.child.pid" 2>/dev/null || true)
  [ -n "$child" ] && kill -TERM -- "-$child" 2>/dev/null || true
  [ -n "$guardian" ] && kill -TERM "$guardian" 2>/dev/null || true
  sleep 5
}
cleanup_experiment() {
  trap - EXIT INT TERM
  stop_registered custom_train || true
  stop_registered target || true
  stop_registered telemetry || true
}
trap cleanup_experiment EXIT INT TERM
wait_health() {
  local port=$1 state_name=$2 deadline=$(( $(date +%s) + 900 ))
  until curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; do
    [ -f runtime/STOP_REQUESTED ] && exit 75
    [ "$(date +%s)" -gt "$deadline" ] && return 1
    sleep 5
  done
}

stage wait_setup
until [ -f artifacts/SETUP_COMPLETE ]; do
  [ -f runtime/STOP_REQUESTED ] && exit 75
  sleep 10
done
"$root/speculators_venv/bin/python" -c 'import datasets,aiohttp' || \
  uv pip install --python "$root/speculators_venv/bin/python" datasets aiohttp
"$root/vllm_venv/bin/python" -c 'import nixl' || \
  uv pip install --python "$root/vllm_venv/bin/python" nixl

stage stop_existing_gpu2_gpu3
for pid in 3659373 4146888; do stop_tree "$pid"; done
sleep 20
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > artifacts/post_service_stop_gpu_processes.csv || true

stage start_telemetry
: > logs/telemetry.log
nohup bash scripts/remote_guardian.sh telemetry bash scripts/telemetry.sh > logs/telemetry_launcher.log 2>&1 < /dev/null &
echo $! > runtime/telemetry.guardian.pid

stage start_target
: > logs/target.log
nohup bash scripts/remote_guardian.sh target \
  taskset -c 0-23 env CUDA_VISIBLE_DEVICES=2 HF_HOME="$HF_HOME" XDG_CACHE_HOME="$XDG_CACHE_HOME" VLLM_CACHE_ROOT="$VLLM_CACHE_ROOT" TORCHINDUCTOR_CACHE_DIR="$TORCHINDUCTOR_CACHE_DIR" \
  "$root/vllm_venv/bin/python" "$root/speculators/scripts/launch_vllm.py" Qwen/Qwen3-8B -- \
  --port 18800 --gpu-memory-utilization 0.82 --max-model-len 4096 \
  > logs/target_launcher.log 2>&1 < /dev/null &
echo $! > runtime/target.guardian.pid
wait_health 18800 target

stage guardian_recovery_test
attempt_before=$(python3 -c 'import json; print(json.load(open("runtime/target.json"))["attempt"])')
target_child=$(cat runtime/target.child.pid)
kill -KILL -- "-$target_child"
deadline=$(( $(date +%s) + 900 ))
while :; do
  attempt_after=$(python3 -c 'import json; print(json.load(open("runtime/target.json"))["attempt"])' 2>/dev/null || echo 0)
  if [ "$attempt_after" -gt "$attempt_before" ] && curl -fsS http://127.0.0.1:18800/health >/dev/null 2>&1; then break; fi
  [ "$(date +%s)" -gt "$deadline" ] && exit 1
  sleep 5
done
printf '{"killed_child":%s,"attempt_before":%s,"attempt_after":%s,"recovered":true,"verified_at":"%s"}\n' \
  "$target_child" "$attempt_before" "$attempt_after" "$(date --iso-8601=seconds)" > artifacts/guardian_recovery_test.json

stage generate_on_policy_math
if [ ! -s data/gsm8k_qwen3_8b.jsonl ]; then
  "$root/speculators_venv/bin/python" scripts/generate_math_onpolicy.py \
    --endpoint http://127.0.0.1:18800/v1 --samples 1000 --concurrency 32 \
    --output data/gsm8k_qwen3_8b.jsonl
fi
sha256sum data/gsm8k_qwen3_8b.jsonl > artifacts/gsm8k_qwen3_8b.sha256

stage prepare_training_data
if [ ! -f data/gsm8k_prepared/state.json ]; then
  "$root/speculators_venv/bin/speculators" prepare-data \
    --model Qwen/Qwen3-8B --data data/gsm8k_qwen3_8b.jsonl \
    --render-endpoint http://127.0.0.1:18800 --output data/gsm8k_prepared \
    --max-samples 1000 --seq-length 2048
fi

stage train_custom_eagle3
nohup bash scripts/remote_guardian.sh custom_train bash scripts/run_custom_training.sh \
  > logs/custom_train_launcher.log 2>&1 < /dev/null &
echo $! > runtime/custom_train.guardian.pid
while :; do
  status=$(python3 -c 'import json; print(json.load(open("runtime/custom_train.json"))["status"])' 2>/dev/null || echo starting)
  [ "$status" = complete ] && break
  [ "$status" = failed ] && exit 1
  [ -f runtime/STOP_REQUESTED ] && exit 75
  sleep 15
done
stop_registered target

stage benchmark_baseline
bash scripts/run_benchmark_config.sh baseline_bf16 Qwen/Qwen3-8B

stage benchmark_pretrained
for k in 1 2 3 4; do
  [ -f runtime/STOP_REQUESTED ] && exit 75
  bash scripts/run_benchmark_config.sh "pretrained_eagle3_k${k}" Qwen/Qwen3-8B RedHatAI/Qwen3-8B-speculator.eagle3 "$k"
done

custom=$(find artifacts/custom_eagle3 -type f -name config.json -printf '%h\n' | sort -V | tail -1)
[ -n "$custom" ] || { echo 'No custom checkpoint found' >&2; exit 1; }
printf '%s\n' "$custom" > artifacts/custom_checkpoint_path.txt
stage benchmark_custom
for k in 1 2 3 4; do
  [ -f runtime/STOP_REQUESTED ] && exit 75
  bash scripts/run_benchmark_config.sh "custom_eagle3_k${k}" Qwen/Qwen3-8B "$custom" "$k"
done

stage benchmark_fp8
bash scripts/run_benchmark_config.sh fp8_weights_bf16_kv RedHatAI/Qwen3-8B-FP8-dynamic
bash scripts/run_benchmark_config.sh fp8_weights_fp8_kv RedHatAI/Qwen3-8B-FP8-dynamic '' 0 fp8_e4m3

stage attempt_pd_disaggregation
if [ ! -f runtime/STOP_REQUESTED ]; then
  bash scripts/run_pd_smoke.sh || printf '%s\n' "$(date --iso-8601=seconds) PD smoke failed; see logs/pd_smoke.log" > artifacts/PD_SMOKE_FAILED
fi

stage aggregate
"$root/vllm_venv/bin/python" scripts/aggregate_results.py
touch artifacts/EXPERIMENT_COMPLETE
stage complete
trap - EXIT INT TERM
