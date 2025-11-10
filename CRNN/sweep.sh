#!/usr/bin/env bash
# Multi-GPU Sweep Runner for CRNN
set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
GPUS=(0 1 2 3)  # GPU IDs to use
CONFIG="config.yaml"
ENTITY="subahininadarajh-basel-university"
PROJECT="CRNN-sweep"

echo -e "${BLUE}=================================================${NC}"
echo -e "${BLUE}      CRNN Multi-GPU Sweep Runner${NC}"
echo -e "${BLUE}=================================================${NC}"

# Check if sweep_train.py exists
if [[ ! -f "sweepTrainer.py" ]]; then
    echo -e "${RED}Error: sweepTrainer.py not found${NC}"
    exit 1
fi

# Check if config exists
if [[ ! -f "$CONFIG" ]]; then
    echo -e "${RED}Error: $CONFIG not found${NC}"
    exit 1
fi

# Function to check if wandb is installed
check_wandb() {
    if ! command -v wandb &> /dev/null; then
        echo -e "${RED}Error: wandb not installed${NC}"
        echo -e "${YELLOW}Install with: pip install wandb${NC}"
        exit 1
    fi
}

# Function to initialize sweep
init_sweep() {
    echo -e "${YELLOW}Initializing W&B sweep...${NC}"
    python3 sweepTrainer.py --init --config "$CONFIG"

    echo -e "\n${YELLOW}Enter the sweep ID from above:${NC}"
    read -r SWEEP_ID
    echo "$SWEEP_ID" > .sweep_id
    echo -e "${GREEN}Sweep ID saved to .sweep_id${NC}"
}

# Function to start agents on multiple GPUs
start_agents() {
    if [[ ! -f .sweep_id ]]; then
        echo -e "${RED}Error: No sweep ID found. Run with --init first${NC}"
        exit 1
    fi

    SWEEP_ID=$(cat .sweep_id)
    echo -e "${BLUE}Starting agents for sweep: ${SWEEP_ID}${NC}"

    # Create logs directory
    mkdir -p sweep_logs

    # Start an agent on each GPU
    for gpu in "${GPUS[@]}"; do
        log_file="sweep_logs/gpu${gpu}.log"
        echo -e "${GREEN}Starting agent on GPU ${gpu}...${NC}"

        # Start agent in background
        CUDA_VISIBLE_DEVICES=$gpu \
        CRNN_CONFIG_PATH="$CONFIG" \
        nohup wandb agent "${ENTITY}/${PROJECT}/${SWEEP_ID}" \
            > "$log_file" 2>&1 &

        pid=$!
        echo "$pid" > "sweep_logs/gpu${gpu}.pid"
        echo -e "${BLUE}  GPU ${gpu}: PID ${pid} (log: ${log_file})${NC}"

        # Small delay to avoid overwhelming the system
        sleep 2
    done

    echo -e "\n${GREEN}All agents started!${NC}"
    echo -e "${YELLOW}View logs: tail -f sweep_logs/*.log${NC}"
    echo -e "${YELLOW}View results: https://wandb.ai/${ENTITY}/${PROJECT}${NC}"
    echo -e "${YELLOW}Stop agents: ./sweep.sh --stop${NC}"
}

# Function to stop all agents
stop_agents() {
    echo -e "${YELLOW}Stopping all sweep agents...${NC}"

    if [[ ! -d sweep_logs ]]; then
        echo -e "${YELLOW}No agents found${NC}"
        exit 0
    fi

    for pid_file in sweep_logs/*.pid; do
        if [[ -f "$pid_file" ]]; then
            pid=$(cat "$pid_file")
            gpu=$(basename "$pid_file" .pid)

            if kill -0 "$pid" 2>/dev/null; then
                echo -e "${BLUE}Stopping ${gpu} (PID ${pid})...${NC}"
                kill "$pid" 2>/dev/null || true
                rm "$pid_file"
            else
                echo -e "${YELLOW}${gpu} (PID ${pid}) not running${NC}"
                rm "$pid_file"
            fi
        fi
    done

    echo -e "${GREEN}All agents stopped${NC}"
}

# Function to show status
show_status() {
    echo -e "${BLUE}Sweep Agent Status:${NC}\n"

    if [[ ! -d sweep_logs ]]; then
        echo -e "${YELLOW}No agents found${NC}"
        exit 0
    fi

    for pid_file in sweep_logs/*.pid; do
        if [[ -f "$pid_file" ]]; then
            pid=$(cat "$pid_file")
            gpu=$(basename "$pid_file" .pid)

            if kill -0 "$pid" 2>/dev/null; then
                echo -e "${GREEN}✓ ${gpu} (PID ${pid}) - RUNNING${NC}"
            else
                echo -e "${RED}✗ ${gpu} (PID ${pid}) - STOPPED${NC}"
                rm "$pid_file"
            fi
        fi
    done

    echo ""
    if [[ -f .sweep_id ]]; then
        SWEEP_ID=$(cat .sweep_id)
        echo -e "${YELLOW}Sweep ID: ${SWEEP_ID}${NC}"
        echo -e "${YELLOW}View: https://wandb.ai/${ENTITY}/${PROJECT}/sweeps/${SWEEP_ID}${NC}"
    fi
}

# Function to show help
show_help() {
    cat << EOF
Usage: ./run_sweep.sh [COMMAND]

Commands:
  --init          Initialize a new sweep
  --start         Start agents on all GPUs
  --stop          Stop all running agents
  --status        Show agent status
  --restart       Stop and restart all agents
  --help          Show this help message

Examples:
  # First time setup:
  ./run_sweep.sh --init
  ./run_sweep.sh --start

  # Check status:
  ./run_sweep.sh --status

  # View logs:
  tail -f sweep_logs/gpu0.log

  # Stop agents:
  ./run_sweep.sh --stop
EOF
}

# Main
check_wandb

case "${1:-}" in
    --init)
        init_sweep
        ;;
    --start)
        start_agents
        ;;
    --stop)
        stop_agents
        ;;
    --status)
        show_status
        ;;
    --restart)
        stop_agents
        sleep 2
        start_agents
        ;;
    --help|"")
        show_help
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        show_help
        exit 1
        ;;
esac