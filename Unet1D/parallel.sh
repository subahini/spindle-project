#!/usr/bin/env bash
# Parallel UNet1D launcher — GPU-aware with reservations
set -Eeuo pipefail

### --- activate the right environment (mamba/conda) ---
if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
fi
conda activate spindle || { echo "[ERR] conda activate spindle failed"; exit 1; }
export MPLBACKEND=Agg

### --- user knobs ---
BASE_CONFIG="config.yaml"
GPUS=( 5 6 7 )              # physical GPU ids to consider
MAX_CONCURRENT=3          # <= number of GPUs
FREE_MEM_MIN=2000         # MiB free to consider a GPU available
TIMEOUT_SECS=7200         # per-run timeout (seconds)

# Optional: W&B key via env (safer than in YAML)
# export WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

### --- colors ---
RED='\033[0;31m'; GREEN='\033[0;32m'; BLUE='\033[0;34m'; YELLOW='\033[1;33m'; NC='\033[0m'

echo -e "${BLUE}===============================================${NC}"
echo -e "${BLUE}UNet1D GPU Parallel Hyperparameter Search${NC}"
echo -e "${BLUE}===============================================${NC}"

[[ -f "$BASE_CONFIG" ]] || { echo -e "${RED}Error: $BASE_CONFIG not found${NC}"; exit 1; }
[[ -f "unet1d.py"     ]] || { echo -e "${RED}Error: unet1d.py not found${NC}"; exit 1; }

echo -e "${YELLOW}Python: $(python -c 'import sys; print(sys.executable)')${NC}"
echo -e "${YELLOW}Checking GPU availability...${NC}"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=index,name,memory.free,memory.total --format=csv,noheader,nounits
else
  echo -e "${YELLOW}Warning: nvidia-smi not found; assuming GPUs exist${NC}"
fi

mkdir -p results_parallel checkpoints logs_parallel results

### --- configs (loss, lr, sampler, name) ---

declare -a CONFIGS=(
  # lr = 5e-5
  "bce,0.00005,normal,bce_normal_5e-5"
  "bce,0.00005,weighted,bce_weighted_5e-5"
  "bce,0.00005,undersample,bce_under_5e-5"

  "focal,0.00005,normal,focal_normal_5e-5"
  "focal,0.00005,weighted,focal_weighted_5e-5"
  "focal,0.00005,undersample,focal_under_5e-5"

  "dice,0.00005,normal,dice_normal_5e-5"
  "dice,0.00005,weighted,dice_weighted_5e-5"
  "dice,0.00005,undersample,dice_under_5e-5"

  # lr = 5e-4
  "focal,0.0005,normal,focal_normal_5e-4"
  "focal,0.0005,weighted,focal_weighted_5e-4"
  "focal,0.0005,undersample,focal_under_5e-4"

  # lr = 3e-4
  "bce,0.0003,normal,bce_normal_3e-4"
  "weighted_bce,0.0003,normal,wbce_normal_3e-4"
  "weighted_bce,0.0003,weighted,wbce_weighted_3e-4"
  "weighted_bce,0.0003,undersample,wbce_under_3e-4"
  "dice,0.0003,normal,dice_normal_3e-4"
)

### --- reservations & jobs tracking ---
declare -A RESERVED=()      # GPU id -> reserved flag
declare -A running_jobs=()  # PID -> GPU id

### --- helpers ---
is_gpu_free() {
  local id="$1"
  if command -v nvidia-smi >/dev/null; then
    local free
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$id" 2>/dev/null | tr -d ' ' || echo 0)
    [[ "${free:-0}" -ge $FREE_MEM_MIN ]]
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
    echo -e "${YELLOW}[INFO] All GPUs busy; waiting 20s...${NC}"
    sleep 20
  done
}

create_config() {
  local loss="$1" lr="$2" sampler="$3" name="$4" gpu_id="$5"
  local cfg="results_parallel/config_${name}.yaml"
  cp "$BASE_CONFIG" "$cfg"
  python - "$cfg" "$loss" "$lr" "$sampler" "$name" "$gpu_id" <<'PY'
import sys, yaml
cfg, loss, lr, sampler, name, gpu_id = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4], sys.argv[5], int(sys.argv[6])
with open(cfg,'r') as f: data = yaml.safe_load(f) or {}
data.setdefault('trainer',{}); data.setdefault('paths',{}); data.setdefault('logging',{}); data.setdefault('eval',{}); data['logging'].setdefault('wandb',{})
# UNet1D expects these under trainer + logging:
data['trainer']['device']='cuda:0'    # CUDA_VISIBLE_DEVICES selects physical GPU; this is the logical index
data['trainer']['loss']=loss
data['trainer']['lr']=lr
data['trainer']['sampler']=sampler
# optional eval tweak; keep if your script reads it
data['eval']['threshold_mode']=data['eval'].get('threshold_mode','sweep')
# output dirs (adjust if your script uses different keys)
data['paths']['checkpoint_dir']=f'./checkpoints/{name}'
data['paths']['results_dir']=f'./results/{name}'
# W&B project (respect existing; else default)
data['logging']['wandb']['project']=data['logging'].get('wandb',{}).get('project','spindle-comparison')
data['logging']['run_name']=f'gpu{gpu_id}_{name}'
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
  if timeout "$TIMEOUT_SECS" python unet1d.py --config "$cfg_file" >"$log_file" 2>&1; then
    local t1; t1=$(date +%s); local mins=$(( (t1-t0)/60 ))
    # best F1 scraping (adjust grep if your log uses a different format)
    local best_f1
    best_f1=$(grep -i -o 'f1=[0-9]*\.[0-9]*' "$log_file" | sed -E 's/.*f1=([0-9.]+)/\1/' | sort -nr | head -1)
    [[ -z "$best_f1" ]] && best_f1="N/A"
    echo -e "${GREEN}[GPU $gpu_id] SUCCESS: $name | F1=$best_f1 | ${mins}min${NC}"
    echo "GPU$gpu_id,$name,$loss,$lr,$sampler,SUCCESS,$best_f1,${mins}min" >> results_parallel/summary.csv
  else
    local t1; t1=$(date +%s); local mins=$(( (t1-t0)/60 ))
    echo -e "${RED}[GPU $gpu_id] FAILED : $name | ${mins}min${NC}"
    echo "GPU$gpu_id,$name,$loss,$lr,$sampler,FAILED,N/A,${mins}min" >> results_parallel/summary.csv
  fi
}

### --- init summary ---
echo "GPU,Name,Loss,LearningRate,Sampler,Status,BestF1,Duration" > results_parallel/summary.csv
echo -e "${YELLOW}Starting parallel training with ${#CONFIGS[@]} configurations...${NC}"
job_count=0

### --- main loop ---
for config in "${CONFIGS[@]}"; do
  # respect MAX_CONCURRENT
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
  echo -e "${BLUE}[INFO] Started job $job_count/${#CONFIGS[@]} on GPU $gpu_id${NC}"
  sleep 2
done

### --- wait for remaining jobs ---
echo -e "${YELLOW}Waiting for all jobs to complete...${NC}"
for pid in "${!running_jobs[@]}"; do
  wait "$pid" || true
  gid="${running_jobs[$pid]}"; unset 'running_jobs[$pid]'
  unset 'RESERVED[$gid]'
  echo -e "${BLUE}[INFO] Final job on GPU $gid completed${NC}"
done

echo -e "${GREEN}===============================================${NC}"
echo -e "${GREEN}All jobs completed!${NC}"
echo -e "${GREEN}===============================================${NC}"

### --- summary table ---
if [[ -f results_parallel/summary.csv ]]; then
  echo -e "${YELLOW}Results Summary (Top 10):${NC}"
  {
    head -1 results_parallel/summary.csv
    tail -n +2 results_parallel/summary.csv | grep SUCCESS | sort -t, -k7 -nr | head -10
  } | column -t -s ','
  failed_count=$(grep -c FAILED results_parallel/summary.csv 2>/dev/null || echo 0)
  success_count=$(grep -c SUCCESS results_parallel/summary.csv 2>/dev/null || echo 0)
  echo -e "${BLUE}Total: ${success_count} succeeded, ${failed_count} failed${NC}"
fi
