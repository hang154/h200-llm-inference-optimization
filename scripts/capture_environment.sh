#!/usr/bin/env bash
set -euo pipefail
root=${EXPERIMENT_ROOT:-/home/work/timecore-ai/h200-inference-optimization-20260929}
out="$root/artifacts/environment.txt"
{
  date --iso-8601=seconds
  uname -a
  nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader -i 2,3
  "$root/vllm_venv/bin/python" - <<'PY'
import importlib.metadata as m
for package in ['torch','vllm','speculators','transformers','datasets','nixl']:
    try: print(f'{package}=={m.version(package)}')
    except m.PackageNotFoundError: print(f'{package}=not-installed')
PY
  printf 'speculators_commit='
  git -C "$root/speculators" rev-parse HEAD
  printf 'script_tree_sha256='
  find "$root/scripts" -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1
} > "$out.tmp"
mv "$out.tmp" "$out"
