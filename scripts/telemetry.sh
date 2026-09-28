#!/usr/bin/env bash
set -euo pipefail
root=${EXPERIMENT_ROOT:-/home/work/timecore-ai/h200-inference-optimization-20260929}
out="$root/telemetry/gpu_2_3.csv"
if [ ! -s "$out" ]; then
  printf 'timestamp,index,utilization_gpu_pct,memory_used_mib,memory_total_mib,power_w,temperature_c\n' > "$out"
fi
while [ ! -f "$root/runtime/STOP_REQUESTED" ]; do
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits -i 2,3 >> "$out"
  sleep 2
done
