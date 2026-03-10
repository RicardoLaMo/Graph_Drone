#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf '%s\n' \
    "Usage: $0 [smoke|full] [--interval-seconds N] [--print-only]" \
    "" \
    "Poll GPU status and launch portfolio lanes once their reserved GPU groups are free." \
    "Default poll interval: 900 seconds (15 minutes)." \
    "Lane A GPUs: GRAPHDRONE_LANE_A_GPUS=3,2,1,0" \
    "Lane B GPUs: GRAPHDRONE_LANE_B_GPUS=7,6,5,4"
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)"
cd "$repo_root"

mode="full"
interval_seconds="${GRAPHDRONE_PORTFOLIO_MONITOR_INTERVAL_SECONDS:-900}"
memory_threshold="${GRAPHDRONE_PORTFOLIO_GPU_MEMORY_THRESHOLD_MIB:-4096}"
util_threshold="${GRAPHDRONE_PORTFOLIO_GPU_UTIL_THRESHOLD:-10}"
lane_a_gpus="${GRAPHDRONE_LANE_A_GPUS:-3,2,1,0}"
lane_b_gpus="${GRAPHDRONE_LANE_B_GPUS:-7,6,5,4}"
print_only=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    smoke|full)
      mode="$1"
      shift
      ;;
    --interval-seconds)
      interval_seconds="${2:-}"
      shift 2
      ;;
    --print-only)
      print_only=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

timestamp="${GRAPHDRONE_PORTFOLIO_MONITOR_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
run_root="$repo_root/experiments/openml_regression_benchmark/reports_lane_runs/monitor_${mode}_$timestamp"
state_dir="$run_root/state"
mkdir -p "$state_dir"
monitor_log="$run_root/monitor.log"
status_file="$run_root/status.txt"
pid_file="$run_root/monitor.pid"

lane_a_script="$repo_root/experiments/openml_regression_benchmark/scripts/run_lane_a_small_datasets.sh"
lane_b_script="$repo_root/experiments/openml_regression_benchmark/scripts/run_lane_b_diamonds.sh"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$monitor_log"
}

normalize_csv() {
  printf '%s' "$1" | tr -d ' '
}

gpu_group_ready() {
  local gpu_csv
  local gpu
  declare -A gpu_mem=()
  declare -A gpu_util=()

  gpu_csv="$(normalize_csv "$1")"
  while IFS=',' read -r idx mem util; do
    idx="${idx// /}"
    mem="${mem// /}"
    util="${util// /}"
    gpu_mem["$idx"]="$mem"
    gpu_util["$idx"]="$util"
  done < <(
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
  )

  IFS=',' read -r -a gpu_ids <<< "$gpu_csv"
  for gpu in "${gpu_ids[@]}"; do
    if [[ -z "${gpu_mem[$gpu]:-}" ]]; then
      return 1
    fi
    if (( gpu_mem[$gpu] > memory_threshold || gpu_util[$gpu] > util_threshold )); then
      return 1
    fi
  done
  return 0
}

write_status() {
  {
    printf 'mode=%s\n' "$mode"
    printf 'interval_seconds=%s\n' "$interval_seconds"
    printf 'memory_threshold_mib=%s\n' "$memory_threshold"
    printf 'util_threshold=%s\n' "$util_threshold"
    printf 'lane_a_gpus=%s\n' "$lane_a_gpus"
    printf 'lane_b_gpus=%s\n' "$lane_b_gpus"
    printf 'lane_a_state=%s\n' "${lane_a_state:-pending}"
    printf 'lane_b_state=%s\n' "${lane_b_state:-pending}"
    printf 'lane_a_pid=%s\n' "${lane_a_pid:-}"
    printf 'lane_b_pid=%s\n' "${lane_b_pid:-}"
    printf 'run_root=%s\n' "$run_root"
  } > "$status_file"
}

launch_lane() {
  local __pid_var="$1"
  shift
  local lane_script="$1"
  local lane_log="$2"
  shift 2

  "$lane_script" "$@" >"$lane_log" 2>&1 &
  printf -v "$__pid_var" '%s' "$!"
}

printf '%s\n' "$$" > "$pid_file"
lane_a_state="pending"
lane_b_state="pending"
lane_a_pid=""
lane_b_pid=""
write_status

log "monitor started"
log "lane-a GPUs=$lane_a_gpus lane-b GPUs=$lane_b_gpus interval=${interval_seconds}s"
log "run_root=$run_root"

if [[ "$print_only" -eq 1 ]]; then
  log "print-only mode: no launches will occur"
fi

while true; do
  write_status

  if [[ "$lane_a_state" == "pending" ]] && gpu_group_ready "$lane_a_gpus"; then
    if [[ "$print_only" -eq 1 ]]; then
      log "lane-a would launch now"
      lane_a_state="print-only"
    else
      lane_a_log="$run_root/lane_a.log"
      log "launching lane-a: $lane_a_script $mode"
      launch_lane lane_a_pid "$lane_a_script" "$lane_a_log" "$mode"
      lane_a_state="running"
      printf '%s\n' "$lane_a_pid" > "$state_dir/lane_a.pid"
      log "lane-a launched pid=$lane_a_pid log=$lane_a_log"
    fi
  fi

  if [[ "$lane_b_state" == "pending" ]] && gpu_group_ready "$lane_b_gpus"; then
    if [[ "$print_only" -eq 1 ]]; then
      log "lane-b would launch now"
      lane_b_state="print-only"
    else
      lane_b_log="$run_root/lane_b.log"
      log "launching lane-b: $lane_b_script $mode"
      launch_lane lane_b_pid "$lane_b_script" "$lane_b_log" "$mode"
      lane_b_state="running"
      printf '%s\n' "$lane_b_pid" > "$state_dir/lane_b.pid"
      log "lane-b launched pid=$lane_b_pid log=$lane_b_log"
    fi
  fi

  if [[ "$lane_a_state" == "running" ]] && ! kill -0 "$lane_a_pid" 2>/dev/null; then
    if wait "$lane_a_pid"; then
      lane_a_state="done"
      log "lane-a completed successfully"
    else
      lane_a_state="failed"
      log "lane-a failed"
    fi
  fi

  if [[ "$lane_b_state" == "running" ]] && ! kill -0 "$lane_b_pid" 2>/dev/null; then
    if wait "$lane_b_pid"; then
      lane_b_state="done"
      log "lane-b completed successfully"
    else
      lane_b_state="failed"
      log "lane-b failed"
    fi
  fi

  write_status
  if [[ "$lane_a_state" =~ ^(done|failed|print-only)$ ]] && [[ "$lane_b_state" =~ ^(done|failed|print-only)$ ]]; then
    log "monitor finished"
    exit 0
  fi

  log "sleeping ${interval_seconds}s before next GPU check"
  sleep "$interval_seconds"
done
