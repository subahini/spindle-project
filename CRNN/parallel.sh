#!/usr/bin/env bash
# Parallel CRNN launcher – GPU-aware with reservations
set -Eeuo pipefail

# ---- user knobs ----
BASE_CONFIG="config.yaml"
GPUS=( 0 1 2 3 )            # physical GPU ids to consider
MAX_CONCURRENT=4          # max parallel runs (<= number of GPUs)
FREE_MEM_MIN=2000         # MiB required free to consider a GPU free
TIMEOUT_SECS=7200         # per-run timeout (seconds)

# ---- colors ----
RED='\033[0;31m'; GREEN='\033[0;32m'; BLUE='\033[0;34m'; YELLOW='\033[1;33m'; NC='\033[0m'

echo -e "${BLUE}===============================================${NC}"
echo -e "${BLUE}CRNN GPU Parallel Hyperparameter Search${NC}"
echo -e "${BLUE}===============================================${NC}"

[[ -f "$BASE_CONFIG" ]] || { echo -e "${RED}Error: $BASE_CONFIG not found${NC}"; exit 1; }
[[ -f "Crnn.py"     ]] || { echo -e "${RED}Error: Crnn.py not found${NC}"; exit 1; }

echo -e "${YELLOW}Checking GPU availability...${NC}"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=index,name,memory.free,memory.total --format=csv,noheader,nounits
else
  echo -e "${YELLOW}Warning: nvidia-smi not found; assuming GPUs exist${NC}"
fi

mkdir -p results_parallel checkpoints logs_parallel

# ---- configs (loss,lr,sampler,name) ----
declare -a CONFIGS=(
  "bce,0.0001,normal,bce_lr1e4_normal"
  "bce,0.0003,normal,bce_lr3e4_normal"
  "bce,0.0001,undersample,bce_lr1e4_under"
  "bce,0.0003,undersample,bce_lr3e4_under"
  "weighted_bce,0.0001,normal,wbce_lr1e4_normal"
  "weighted_bce,0.0003,normal,wbce_lr3e4_normal"
  "weighted_bce,0.0001,undersample,wbce_lr1e4_under"
  "weighted_bce,0.0003,undersample,wbce_lr3e4_under"
  "weighted_bce,0.0001,weighted,wbce_lr1e4_weighted"
  "weighted_bce,0.0003,weighted,wbce_lr3e4_weighted"
  "focal,0.0001,normal,focal_lr1e4_normal"
  "focal,0.0003,normal,focal_lr3e4_normal"
  "focal,0.0005,normal,focal_lr5e4_normal"
  "focal,0.0001,undersample,focal_lr1e4_under"
  "focal,0.0003,undersample,focal_lr3e4_under"
  "focal,0.0005,undersample,focal_lr5e4_under"
  "focal,0.0001,weighted,focal_lr1e4_weighted"
  "focal,0.0003,weighted,focal_lr3e4_weighted"
  "dice,0.0001,normal,dice_lr1e4_normal"
  "dice,0.0003,normal,dice_lr3e4_normal"
  "dice,0.0001,undersample,dice_lr1e4_under"
  "dice,0.0003,undersample,dice_lr3e4_under"
)

# ---- reservations & jobs tracking ----
declare -A RESERVED=()      # GPU id -> reserved
declare -A running_jobs=()  # PID -> GPU id

# ---- helpers ----
is_gpu_free() {
  local id="$1"
  if command -v nvidia-smi >/dev/null; then
    local free
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$id" | tr -d ' ')
    [[ "${free:-0}" -gt $FREE_MEM_MIN ]]
  else
    return 0
  fi
}

wait_for_gpu() {
  while true; do
    for id in "${GPUS[@]}"; do
      [[ -n "${RESERVED[$id]:-}" ]] && continue
      if is_gpu_free "$id"; then
        echo "$id"; return 0
      fi
    done
    echo -e "${YELLOW}All GPUs busy; waiting 20s...${NC}"
    sleep 20
  done
}

create_config() {
  local loss="$1" lr="$2" sampler="$3" name="$4" gpu_id="$5"
  local cfg="results_parallel/config_${name}.yaml"
  cp "$BASE_CONFIG" "$cfg"
  python3 - "$cfg" "$loss" "$lr" "$sampler" "$name" "$gpu_id" <<'PY'
import sys, yaml
cfg, loss, lr, sampler, name, gpu_id = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4], sys.argv[5], int(sys.argv[6])
with open(cfg,'r') as f: data=yaml.safe_load(f) or {}

# Ensure all necessary sections exist
data.setdefault('project',{})
data.setdefault('trainer',{})
data.setdefault('paths',{})
data.setdefault('logging',{})
data.setdefault('eval',{})
data.setdefault('loss',{})
data['logging'].setdefault('wandb',{})

# Set device and hyperparameters
data['project']['device']='cuda'
data['trainer']['loss']=loss
data['trainer']['lr']=lr
data['trainer']['sampler']=sampler

# Set evaluation mode
data['eval']['n_thresholds']=data['eval'].get('n_thresholds',101)
data['eval']['metric_for_best']=data['eval'].get('metric_for_best','f1')

# Set paths
data['paths']['out_dir']=f'./checkpoints/{name}'

# W&B configuration
data['logging']['wandb']['enabled']=True
data['logging']['wandb']['project']='CRNN-no-se'
data['logging']['wandb']['entity']=data['logging'].get('wandb',{}).get('entity','subahininadarajh-basel-university')
data['logging']['wandb']['name']=f'gpu{gpu_id}_{name}'
data['logging']['wandb']['tags']=['spindle-detection','CRNN','grid-search',f'gpu:{gpu_id}',f'loss:{loss}',f'lr:{lr}',f'sampler:{sampler}']

# Save config
with open(cfg,'w') as f: yaml.safe_dump(data,f, sort_keys=False)
PY
  echo "$cfg"
}

run_job() {
  local config_params="$1" gpu_id="$2"
  IFS=',' read -r loss lr sampler name <<< "$config_params"
  echo -e "${GREEN}[GPU $gpu_id] Starting: $name (loss=$loss, lr=$lr, sampler=$sampler)${NC}"
  local cfg_file; cfg_file=$(create_config "$loss" "$lr" "$sampler" "$name" "$gpu_id")
  local log_file="logs_parallel/${name}_gpu${gpu_id}.log"
  export CUDA_VISIBLE_DEVICES=$gpu_id
  local t0; t0=$(date +%s)
  if timeout $TIMEOUT_SECS python3 Crnn.py --config "$cfg_file" >"$log_file" 2>&1; then
    local t1; t1=$(date +%s); local mins=$(( (t1-t0)/60 ))
    # Extract best F1 from log
    local best_f1 best_roc_auc best_pr_auc
    best_f1=$(grep -oP 'F1=\K[0-9.]+' "$log_file" | sort -nr | head -1)
    best_roc_auc=$(grep -oP 'ROC-AUC=\K[0-9.]+' "$log_file" | sort -nr | head -1)
    best_pr_auc=$(grep -oP 'PR-AUC=\K[0-9.]+' "$log_file" | sort -nr | head -1)
    [[ -z "$best_f1" ]] && best_f1="N/A"
    [[ -z "$best_roc_auc" ]] && best_roc_auc="N/A"
    [[ -z "$best_pr_auc" ]] && best_pr_auc="N/A"
    echo -e "${GREEN}[GPU $gpu_id] SUCCESS: $name | F1=$best_f1 ROC-AUC=$best_roc_auc PR-AUC=$best_pr_auc | ${mins}min${NC}"
    echo "GPU$gpu_id,$name,$loss,$lr,$sampler,SUCCESS,$best_f1,$best_roc_auc,$best_pr_auc,${mins}min" >> results_parallel/summary.csv
  else
    local t1; t1=$(date +%s); local mins=$(( (t1-t0)/60 ))
    echo -e "${RED}[GPU $gpu_id] FAILED : $name | ${mins}min${NC}"
    echo "GPU$gpu_id,$name,$loss,$lr,$sampler,FAILED,N/A,N/A,N/A,${mins}min" >> results_parallel/summary.csv
  fi
}

# ---- init summary ----
echo "GPU,Name,Loss,LearningRate,Sampler,Status,BestF1,BestROC-AUC,BestPR-AUC,Duration" > results_parallel/summary.csv
echo -e "${YELLOW}Starting parallel CRNN training with ${#CONFIGS[@]} configurations...${NC}"
job_count=0

# ---- main loop ----
for config in "${CONFIGS[@]}"; do
  while ((${#running_jobs[@]} >= MAX_CONCURRENT)); do
    for pid in "${!running_jobs[@]}"; do
      if ! kill -0 "$pid" 2>/dev/null; then
        wait "$pid" || true
        freed="${running_jobs[$pid]}"; unset 'running_jobs[$pid]'
        unset 'RESERVED[$freed]'
        echo -e "${BLUE}[INFO] GPU $freed freed up${NC}"
      fi
    done
    ((${#running_jobs[@]} < MAX_CONCURRENT)) || sleep 5
  done
  
  gpu_id="$(wait_for_gpu)"
  RESERVED[$gpu_id]=1
  run_job "$config" "$gpu_id" &
  pid=$!
  running_jobs[$pid]=$gpu_id
  job_count=$((job_count+1))
  echo -e "${BLUE}[INFO] Started job $job_count/${#CONFIGS[@]} on GPU $gpu_id (PID=$pid)${NC}"
  sleep 2
done

# ---- wait for remaining jobs ----
echo -e "${YELLOW}Waiting for all jobs to complete...${NC}"
for pid in "${!running_jobs[@]}"; do
  wait "$pid" || true
  gid="${running_jobs[$pid]}"; unset 'running_jobs[$pid]'
  unset 'RESERVED[$gid]'
  echo -e "${BLUE}[INFO] Final job on GPU $gid completed${NC}"
done

echo -e "${GREEN}===============================================${NC}"
echo -e "${GREEN}All CRNN jobs completed!${NC}"
echo -e "${GREEN}===============================================${NC}"

# ---- summary table ----
if [[ -f results_parallel/summary.csv ]]; then
  echo -e "${YELLOW}Results Summary (Top 10 by F1):${NC}"
  {
    head -1 results_parallel/summary.csv
    tail -n +2 results_parallel/summary.csv | grep SUCCESS | sort -t, -k7 -nr | head -10
  } | column -t -s ','
  
  failed_count=$(grep -c FAILED results_parallel/summary.csv 2>/dev/null || echo 0)
  success_count=$(grep -c SUCCESS results_parallel/summary.csv 2>/dev/null || echo 0)
  echo -e "${BLUE}Total: ${success_count} succeeded, ${failed_count} failed${NC}"
  echo -e "${YELLOW}Full results saved to: results_parallel/summary.csv${NC}"
  echo -e "${YELLOW}Check W&B project: https://wandb.ai/subahininadarajh-basel-university/CRNN-GridSearch-GPU${NC}"
fi
