#!/usr/bin/env bash
# Cross-Channel Generalization Experiment Runner
set -Eeuo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
CONFIG="config.yaml"
GPUS=(0 1 2 3)  # Available GPUs
OUTPUT_DIR="./results_cross_channel"

# Training configurations
# Format: "train_channel,test_channels"
EXPERIMENTS=(
    "C3,C4 F3 F4"
    "C4,C3 F3 F4"
    "F3,C3 C4 F4"
    "F4,C3 C4 F3"
)

echo -e "${BLUE}=================================================${NC}"
echo -e "${BLUE}   Cross-Channel Generalization Experiments${NC}"
echo -e "${BLUE}=================================================${NC}"

# Check files
[[ -f "$CONFIG" ]] || { echo -e "${RED}Error: $CONFIG not found${NC}"; exit 1; }
[[ -f "cross_channel_train.py" ]] || { echo -e "${RED}Error: cross_channel_train.py not found${NC}"; exit 1; }

# Check GPU availability
echo -e "${YELLOW}Checking GPU availability...${NC}"
if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader
else
    echo -e "${YELLOW}Warning: nvidia-smi not found${NC}"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR/logs"
mkdir -p "$OUTPUT_DIR/checkpoints"

# Track running jobs
declare -A running_jobs=()  # PID -> experiment name
declare -A gpu_assignments=()  # PID -> GPU id

# Function to find free GPU
find_free_gpu() {
    for gpu in "${GPUS[@]}"; do
        # Check if GPU is assigned
        local assigned=0
        for pid in "${!gpu_assignments[@]}"; do
            if [[ "${gpu_assignments[$pid]}" == "$gpu" ]]; then
                # Check if process still running
                if kill -0 "$pid" 2>/dev/null; then
                    assigned=1
                    break
                else
                    # Clean up finished job
                    unset 'gpu_assignments[$pid]'
                    unset 'running_jobs[$pid]'
                fi
            fi
        done

        if [[ $assigned -eq 0 ]]; then
            echo "$gpu"
            return 0
        fi
    done
    return 1
}

# Function to wait for a free GPU
wait_for_gpu() {
    while true; do
        if gpu_id=$(find_free_gpu); then
            echo "$gpu_id"
            return 0
        fi
        echo -e "${YELLOW}All GPUs busy, waiting...${NC}"
        sleep 10
    done
}

# Function to run single experiment
run_experiment() {
    local train_ch="$1"
    local test_chs="$2"
    local gpu_id="$3"
    local exp_name="${train_ch}_to_${test_chs// /_}"

    echo -e "${GREEN}[GPU $gpu_id] Starting: Train on $train_ch, test on $test_chs${NC}"

    local log_file="$OUTPUT_DIR/logs/${exp_name}_gpu${gpu_id}.log"
    local out_dir="$OUTPUT_DIR/checkpoints/$exp_name"

    mkdir -p "$out_dir"

    # Run training
    CUDA_VISIBLE_DEVICES=$gpu_id python3 cross_channel_train.py \
        --config "$CONFIG" \
        --train-channel "$train_ch" \
        --test-channels $test_chs \
        --output-dir "$out_dir" \
        > "$log_file" 2>&1

    local exit_code=$?

    if [[ $exit_code -eq 0 ]]; then
        # Extract results
        local results_file="$out_dir/cross_channel_results_${train_ch}.json"
        if [[ -f "$results_file" ]]; then
            echo -e "${GREEN}[GPU $gpu_id] SUCCESS: $exp_name${NC}"

            # Parse and display key results
            if command -v python3 >/dev/null; then
                python3 - "$results_file" <<'PYTHON'
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
train_ch = data["train_channel"]
results = data["results"]
print(f"  Train {train_ch}: F1={results[train_ch]['f1']:.3f}")
for test_ch in data["test_channels"]:
    print(f"  Test {test_ch}: F1={results[test_ch]['f1']:.3f}")
PYTHON
            fi
        fi
    else
        echo -e "${RED}[GPU $gpu_id] FAILED: $exp_name (exit code: $exit_code)${NC}"
    fi

    return $exit_code
}

# Main execution
echo -e "${YELLOW}Starting ${#EXPERIMENTS[@]} cross-channel experiments...${NC}"

job_count=0
for exp in "${EXPERIMENTS[@]}"; do
    IFS=',' read -r train_ch test_chs <<< "$exp"

    # Wait for free GPU
    gpu_id=$(wait_for_gpu)

    # Start experiment in background
    run_experiment "$train_ch" "$test_chs" "$gpu_id" &
    pid=$!

    # Track job
    running_jobs[$pid]="$train_ch -> $test_chs"
    gpu_assignments[$pid]=$gpu_id

    job_count=$((job_count + 1))
    echo -e "${BLUE}[INFO] Started job $job_count/${#EXPERIMENTS[@]} on GPU $gpu_id (PID=$pid)${NC}"

    sleep 2  # Small delay between launches
done

# Wait for all jobs to complete
echo -e "${YELLOW}Waiting for all experiments to complete...${NC}"
for pid in "${!running_jobs[@]}"; do
    wait "$pid" || true
    exp_name="${running_jobs[$pid]}"
    gpu_id="${gpu_assignments[$pid]}"
    echo -e "${BLUE}[INFO] Completed: $exp_name (GPU $gpu_id)${NC}"
done

echo -e "${GREEN}=================================================${NC}"
echo -e "${GREEN}   All experiments completed!${NC}"
echo -e "${GREEN}=================================================${NC}"

# Generate summary report
echo -e "${YELLOW}Generating summary report...${NC}"

summary_file="$OUTPUT_DIR/summary.txt"
{
    echo "=========================================="
    echo "Cross-Channel Generalization Summary"
    echo "=========================================="
    echo ""

    for exp in "${EXPERIMENTS[@]}"; do
        IFS=',' read -r train_ch test_chs <<< "$exp"
        exp_name="${train_ch}_to_${test_chs// /_}"
        results_file="$OUTPUT_DIR/checkpoints/$exp_name/cross_channel_results_${train_ch}.json"

        if [[ -f "$results_file" ]]; then
            echo "Experiment: Train on $train_ch"
            python3 - "$results_file" <<'PYTHON'
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
train_ch = data["train_channel"]
results = data["results"]
print(f"  Training channel ({train_ch}): F1={results[train_ch]['f1']:.3f} ROC-AUC={results[train_ch]['roc_auc']:.3f}")
test_f1s = [results[ch]['f1'] for ch in data['test_channels']]
import numpy as np
print(f"  Test channels: Mean F1={np.mean(test_f1s):.3f} ± {np.std(test_f1s):.3f}")
for test_ch in data['test_channels']:
    print(f"    {test_ch}: F1={results[test_ch]['f1']:.3f} ROC-AUC={results[test_ch]['roc_auc']:.3f}")
print(f"  Generalization gap: {results[train_ch]['f1'] - np.mean(test_f1s):.3f}")
print()
PYTHON
        else
            echo "Experiment: Train on $train_ch - FAILED"
            echo ""
        fi
    done
} | tee "$summary_file"

echo -e "${GREEN}Summary saved to: $summary_file${NC}"
echo -e "${YELLOW}Results directory: $OUTPUT_DIR${NC}"
echo -e "${YELLOW}Logs directory: $OUTPUT_DIR/logs${NC}"