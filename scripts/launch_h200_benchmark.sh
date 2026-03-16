#!/bin/bash
#
# Launch H200 Parallel TabArena Benchmark
#
# This script:
# 1. Activates h200_tabpfn conda environment
# 2. Launches benchmark in tmux session for detached execution
# 3. Monitors GPU utilization and progress
# 4. Provides real-time logging
#
# Usage:
#   bash scripts/launch_h200_benchmark.sh [OPTIONS]
#
# Options:
#   --quick              Quick test (5 datasets, 1 fold)
#   --monitor            Launch GPU monitoring in separate tmux window
#   --no-checkpoint      Disable checkpointing
#   --attach             Attach to tmux session after launch

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default options
QUICK_MODE=false
ENABLE_MONITOR=true
ENABLE_CHECKPOINT=true
ATTACH_SESSION=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --no-monitor)
            ENABLE_MONITOR=false
            shift
            ;;
        --no-checkpoint)
            ENABLE_CHECKPOINT=false
            shift
            ;;
        --attach)
            ATTACH_SESSION=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Session name
SESSION_NAME="h200_benchmark_$(date +%Y%m%d_%H%M%S)"

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}H200 PARALLEL TABARENA BENCHMARK LAUNCHER${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo -e "${RED}ERROR: conda not found${NC}"
    exit 1
fi

# Check if h200_tabpfn environment exists
if ! conda env list | grep -q "h200_tabpfn"; then
    echo -e "${RED}ERROR: h200_tabpfn conda environment not found${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} conda environment verified"

# Check GPU availability
echo -e "${BLUE}Checking GPU availability...${NC}"
GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo 0)
if [ "$GPU_COUNT" -lt 6 ]; then
    echo -e "${YELLOW}WARNING: Expected 6+ GPUs, found $GPU_COUNT${NC}"
fi
echo -e "${GREEN}✓${NC} Found $GPU_COUNT GPUs"

# Create logs directory
mkdir -p "$ROOT_DIR/logs"
mkdir -p "$ROOT_DIR/eval/tabarena_h200_parallel"

# Benchmark command
BENCHMARK_CMD="cd '$ROOT_DIR' && conda run -n h200_tabpfn python scripts/run_tabarena_h200_parallel.py"

if [ "$QUICK_MODE" = true ]; then
    BENCHMARK_CMD="$BENCHMARK_CMD --quick"
    echo -e "${YELLOW}Quick test mode: 5 datasets, 1 fold (~20 minutes)${NC}"
else
    echo -e "${GREEN}Full benchmark mode: 51 datasets, 3 folds (~4-6 hours)${NC}"
fi

if [ "$ENABLE_CHECKPOINT" = false ]; then
    BENCHMARK_CMD="$BENCHMARK_CMD --no-checkpoint"
fi

# Use GPUs 0 and 1 (2-7 are busy) with 2 workers
# GpuMemoryTracker now disabled in adapter to avoid CUDA sync errors
BENCHMARK_CMD="$BENCHMARK_CMD --checkpoint --gpus 0 1 --workers 2"

echo ""
echo -e "${BLUE}Benchmark configuration:${NC}"
echo "  Session name: $SESSION_NAME"
echo "  GPUs: 0, 1 (GPUs 2-7 are busy)"
echo "  Workers: 2 (one per GPU for parallelism)"
echo "  CUDA_LAUNCH_BLOCKING: 1 (synchronous GPU execution)"
echo "  Checkpoint: $ENABLE_CHECKPOINT"
echo "  Log file: $ROOT_DIR/logs/tabarena_h200_parallel.log"
echo ""

# Create tmux session with benchmark
echo -e "${BLUE}Launching benchmark in tmux session...${NC}"

# Kill existing session if it exists
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

# Create new session
tmux new-session -d -s "$SESSION_NAME" -c "$ROOT_DIR"

# Window 1: Benchmark
tmux send-keys -t "$SESSION_NAME" "echo 'Starting H200 Parallel Benchmark...'; $BENCHMARK_CMD" Enter

echo -e "${GREEN}✓${NC} Benchmark launched in tmux session: $SESSION_NAME"

# Window 2: GPU monitoring (if enabled)
if [ "$ENABLE_MONITOR" = true ]; then
    echo -e "${BLUE}Setting up GPU monitoring...${NC}"
    tmux new-window -t "$SESSION_NAME" -n "gpu-monitor" -c "$ROOT_DIR"
    tmux send-keys -t "$SESSION_NAME:gpu-monitor" "
# GPU monitoring loop
echo 'GPU Monitoring (updates every 5 seconds)'
echo '======================================'
while true; do
    clear
    echo 'GPU Status at '$(date '+%Y-%m-%d %H:%M:%S')
    echo '======================================'
    nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits | awk '{printf \"GPU %d: %s | GPU Util: %d%% | Mem: %d/%dMB\n\", \$1, \$2, \$3, \$5, \$6}'
    echo '======================================'
    tail -20 logs/tabarena_h200_parallel.log | grep -E 'Task|Progress|Completed|Failed' || echo 'Waiting for logs...'
    sleep 5
done
    " Enter
    echo -e "${GREEN}✓${NC} GPU monitoring enabled in window: gpu-monitor"
fi

# Window 3: Log monitoring
echo -e "${BLUE}Setting up log monitoring...${NC}"
tmux new-window -t "$SESSION_NAME" -n "logs" -c "$ROOT_DIR"
tmux send-keys -t "$SESSION_NAME:logs" "tail -f logs/tabarena_h200_parallel.log" Enter
echo -e "${GREEN}✓${NC} Log monitoring enabled in window: logs"

echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}✓ Benchmark launching successfully!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo -e "${BLUE}Tmux commands:${NC}"
echo "  List sessions:  tmux list-sessions"
echo "  Attach session: tmux attach-session -t $SESSION_NAME"
echo "  Kill session:   tmux kill-session -t $SESSION_NAME"
echo ""
echo -e "${BLUE}Session windows:${NC}"
echo "  0: benchmark    - Main benchmark execution"
echo "  1: gpu-monitor  - GPU utilization monitoring"
echo "  2: logs         - Real-time log tail"
echo ""
echo -e "${BLUE}Switch between windows:${NC}"
echo "  tmux select-window -t $SESSION_NAME:0  (benchmark)"
echo "  tmux select-window -t $SESSION_NAME:1  (gpu-monitor)"
echo "  tmux select-window -t $SESSION_NAME:2  (logs)"
echo ""

# Attach if requested
if [ "$ATTACH_SESSION" = true ]; then
    echo -e "${YELLOW}Attaching to session...${NC}"
    sleep 2
    tmux attach-session -t "$SESSION_NAME"
else
    echo -e "${YELLOW}Benchmark running in background. To monitor:${NC}"
    echo -e "${YELLOW}  tmux attach-session -t $SESSION_NAME${NC}"
    echo ""
    # Show initial status
    sleep 3
    echo -e "${BLUE}Initial status:${NC}"
    tmux capture-pane -t "$SESSION_NAME:0" -p | tail -10
fi

echo ""
echo -e "${GREEN}Session: $SESSION_NAME${NC}"
