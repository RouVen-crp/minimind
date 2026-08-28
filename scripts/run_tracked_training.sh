#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <stage> <command> [args...]" >&2
  exit 2
fi

stage="$1"
shift

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
trainer_dir="$repo_root/trainer"
log_dir="$repo_root/experiments/logs"
metrics_dir="$repo_root/experiments/metrics"
runtime_dir="$repo_root/experiments/runtime"
mkdir -p "$log_dir" "$metrics_dir" "$runtime_dir"

if pgrep -af 'train_(pretrain|full_sft)\.py' >/dev/null; then
  echo "A MiniMind training process already exists; refusing duplicate launch." >&2
  pgrep -af 'train_(pretrain|full_sft)\.py' >&2
  exit 3
fi

started_at="$(date --iso-8601=seconds)"
run_id="$(date +%Y%m%d-%H%M%S)"
stdout_path="$log_dir/$stage-$run_id.stdout.log"
gpu_path="$metrics_dir/$stage-$run_id.gpu.csv"
runtime_path="$runtime_dir/$stage-$run_id.json"
latest_path="$runtime_dir/$stage-latest.json"
command_text="$(printf '%q ' "$@")"

write_runtime() {
  local status="$1"
  local exit_code="${2:-}"
  local ended_at="${3:-}"
  local duration_seconds="${4:-}"
  python - "$runtime_path" "$latest_path" "$stage" "$run_id" "$status" "$$" \
    "$started_at" "$ended_at" "$duration_seconds" "$exit_code" "$command_text" \
    "${stdout_path#"$repo_root/"}" "${gpu_path#"$repo_root/"}" <<'PY'
import json
import pathlib
import sys

(runtime_path, latest_path, stage, run_id, status, pid, started_at,
 ended_at, duration_seconds, exit_code, command, stdout, gpu_metrics) = sys.argv[1:]
data = {
    "stage": stage,
    "run_id": run_id,
    "status": status,
    "pid": int(pid),
    "started_at": started_at,
    "command": command.strip(),
    "working_directory": "trainer",
    "stdout": stdout,
    "gpu_metrics": gpu_metrics,
}
if ended_at:
    data["ended_at"] = ended_at
if duration_seconds:
    data["duration_seconds"] = float(duration_seconds)
if exit_code:
    data["exit_code"] = int(exit_code)
payload = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
pathlib.Path(runtime_path).write_text(payload, encoding="utf-8")
pathlib.Path(latest_path).write_text(payload, encoding="utf-8")
PY
}

echo 'timestamp,index,name,temperature_c,utilization_gpu_pct,memory_used_mib,memory_total_mib,power_draw_w' > "$gpu_path"
(
  while true; do
    nvidia-smi \
      --query-gpu=timestamp,index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw \
      --format=csv,noheader,nounits >> "$gpu_path" || true
    sleep 60
  done
) &
gpu_pid=$!
trap 'kill "$gpu_pid" 2>/dev/null || true' EXIT

write_runtime running
start_epoch_seconds="$(date +%s)"
cd "$trainer_dir"
set +e
"$@" 2>&1 | tee "$stdout_path"
exit_code=${PIPESTATUS[0]}
set -e
ended_at="$(date --iso-8601=seconds)"
duration_seconds="$(( $(date +%s) - start_epoch_seconds ))"
status=failed
if [[ $exit_code -eq 0 ]]; then
  status=completed
fi
write_runtime "$status" "$exit_code" "$ended_at" "$duration_seconds"
exit "$exit_code"

