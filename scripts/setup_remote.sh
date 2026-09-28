#!/usr/bin/env bash
set -euo pipefail

root=${EXPERIMENT_ROOT:-/home/work/timecore-ai/h200-inference-optimization-20260929}
mkdir -p "$root"/{runtime,artifacts,data,results,telemetry,logs}
cd "$root"
export UV_CACHE_DIR="$root/.cache/uv"
export HF_HOME="$root/.cache/huggingface"
export TMPDIR="$root/.cache/tmp"
mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$TMPDIR"

if [ ! -d speculators/.git ]; then
  git clone --filter=blob:none https://github.com/vllm-project/speculators.git speculators
fi
git -C speculators fetch --depth 1 origin main
git -C speculators checkout --detach FETCH_HEAD
git -C speculators rev-parse HEAD > artifacts/speculators_commit.txt

if [ ! -x speculators_venv/bin/python ]; then
  uv venv --python 3.12 speculators_venv
fi
uv pip install --python speculators_venv/bin/python -e ./speculators datasets aiohttp

if [ ! -x vllm_venv/bin/python ]; then
  uv venv --python 3.12 vllm_venv
fi
uv pip install --python vllm_venv/bin/python 'vllm>=0.27.1' 'guidellm>=0.5.0' pandas pyarrow matplotlib nixl

speculators_venv/bin/python -c 'import importlib.metadata; print(importlib.metadata.version("speculators"))' > artifacts/speculators_version.txt
vllm_venv/bin/python -c 'import torch,vllm; print(torch.__version__); print(vllm.__version__)' > artifacts/vllm_version.txt
python3 - <<'PY'
import hashlib, json, os
from pathlib import Path
root=Path(os.environ.get('EXPERIMENT_ROOT','/home/work/timecore-ai/h200-inference-optimization-20260929'))
fingerprint={}
for name in ('speculators_commit.txt','speculators_version.txt','vllm_version.txt'):
    p=root/'artifacts'/name
    fingerprint[name]=hashlib.sha256(p.read_bytes()).hexdigest()
tmp=root/'artifacts'/'environment_fingerprint.json.tmp'
tmp.write_text(json.dumps(fingerprint,indent=2))
tmp.replace(root/'artifacts'/'environment_fingerprint.json')
PY
touch artifacts/SETUP_COMPLETE
