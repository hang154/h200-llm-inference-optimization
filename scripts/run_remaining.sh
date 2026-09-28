#!/usr/bin/env bash
set -euo pipefail
root=${EXPERIMENT_ROOT:-/home/work/timecore-ai/h200-inference-optimization-20260929}
cd "$root"
state=runtime/remaining_state.json
mark(){ printf '{"stage":"%s","updated_at":"%s"}\n' "$1" "$(date --iso-8601=seconds)" > "$state.tmp"; mv "$state.tmp" "$state"; }
custom=$(cat artifacts/custom_checkpoint_path.txt)
mark custom
for k in 1 2 3 4; do
  [ -f "results/raw/custom_eagle3_k${k}/COMPLETE" ] && continue
  [ -f runtime/STOP_REQUESTED ] && exit 75
  bash scripts/run_benchmark_config.sh "custom_eagle3_k${k}" Qwen/Qwen3-8B "$custom" "$k"
done
mark fp8
[ -f results/raw/fp8_weights_bf16_kv/COMPLETE ] || bash scripts/run_benchmark_config.sh fp8_weights_bf16_kv RedHatAI/Qwen3-8B-FP8-dynamic
[ -f results/raw/fp8_weights_fp8_kv/COMPLETE ] || bash scripts/run_benchmark_config.sh fp8_weights_fp8_kv RedHatAI/Qwen3-8B-FP8-dynamic '' 0 fp8_e4m3
mark wait_domain_suite
while [ ! -f artifacts/DOMAIN_SUITE_COMPLETE ]; do
  [ -f runtime/STOP_REQUESTED ] && exit 75
  sleep 15
done
mark quality_regression
bash scripts/run_quality_config.sh baseline_bf16 Qwen/Qwen3-8B
bash scripts/run_quality_config.sh fp8_weights RedHatAI/Qwen3-8B-FP8-dynamic
mark pd_disaggregation
if [ ! -f runtime/STOP_REQUESTED ]; then
  bash scripts/run_pd_smoke.sh || printf '%s\n' "$(date --iso-8601=seconds) PD smoke failed; see logs/pd_smoke.log" > artifacts/PD_SMOKE_FAILED
fi
mark aggregate
vllm_venv/bin/python scripts/aggregate_results.py
touch artifacts/EXPERIMENT_COMPLETE
mark complete
