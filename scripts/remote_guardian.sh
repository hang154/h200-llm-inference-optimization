#!/usr/bin/env bash
set -u
root=${EXPERIMENT_ROOT:-/home/work/timecore-ai/h200-inference-optimization-20260929}
name=$1
shift
state="$root/runtime/${name}.json"
log="$root/logs/${name}.log"
attempt=0
child=''
cleanup() {
  if [ -n "$child" ] && kill -0 "$child" 2>/dev/null; then
    kill -TERM -- "-$child" 2>/dev/null || true
    sleep 2
    kill -KILL -- "-$child" 2>/dev/null || true
  fi
}
on_signal() {
  trap - INT TERM EXIT
  cleanup
  exit 143
}
trap on_signal INT TERM
trap cleanup EXIT
while (( attempt < 8 )); do
  printf '{"status":"running","attempt":%d,"pid":%d,"updated_at":"%s"}\n' "$attempt" "$$" "$(date --iso-8601=seconds)" > "$state.tmp"
  mv "$state.tmp" "$state"
  setsid "$@" >> "$log" 2>&1 &
  child=$!
  printf '%s\n' "$child" > "$root/runtime/${name}.child.pid"
  while kill -0 "$child" 2>/dev/null; do
    printf '{"status":"running","attempt":%d,"pid":%d,"child_pid":%d,"updated_at":"%s"}\n' "$attempt" "$$" "$child" "$(date --iso-8601=seconds)" > "$state.tmp"
    mv "$state.tmp" "$state"
    sleep 20
  done
  wait "$child"
  rc=$?
  if [ "$rc" -eq 0 ]; then
    printf '{"status":"complete","attempt":%d,"updated_at":"%s"}\n' "$attempt" "$(date --iso-8601=seconds)" > "$state.tmp"
    mv "$state.tmp" "$state"
    trap - INT TERM EXIT
    exit 0
  fi
  attempt=$((attempt + 1))
  printf '{"status":"cooldown","attempt":%d,"last_exit":%d,"updated_at":"%s"}\n' "$attempt" "$rc" "$(date --iso-8601=seconds)" > "$state.tmp"
  mv "$state.tmp" "$state"
  sleep $(( 5 * (2 ** (attempt < 6 ? attempt : 6)) ))
done
printf '{"status":"failed","attempt":%d,"updated_at":"%s"}\n' "$attempt" "$(date --iso-8601=seconds)" > "$state"
trap - INT TERM EXIT
exit 1
