#!/usr/bin/env bash
set -euo pipefail
root=${EXPERIMENT_ROOT:-/home/work/timecore-ai/h200-inference-optimization-20260929}
cd "$root"
state=runtime/domain_suite_state.json
mark(){ printf '{"stage":"%s","updated_at":"%s"}\n' "$1" "$(date --iso-8601=seconds)" > "$state.tmp"; mv "$state.tmp" "$state"; }
mark baseline
bash scripts/run_domain_config.sh baseline_bf16 Qwen/Qwen3-8B
mark pretrained
for k in 1 2 3 4; do
  [ -f runtime/STOP_REQUESTED ] && exit 75
  bash scripts/run_domain_config.sh "pretrained_eagle3_k${k}" Qwen/Qwen3-8B RedHatAI/Qwen3-8B-speculator.eagle3 "$k"
done
custom=$(cat artifacts/custom_checkpoint_path.txt 2>/dev/null || find artifacts/custom_eagle3 -type f -name config.json -printf '%h\n' | sort -V | tail -1)
[ -n "$custom" ]
printf '%s\n' "$custom" > artifacts/custom_checkpoint_path.txt
mark custom
for k in 1 2 3 4; do
  [ -f runtime/STOP_REQUESTED ] && exit 75
  bash scripts/run_domain_config.sh "custom_eagle3_k${k}" Qwen/Qwen3-8B "$custom" "$k"
done
mark complete
touch artifacts/DOMAIN_SUITE_COMPLETE
