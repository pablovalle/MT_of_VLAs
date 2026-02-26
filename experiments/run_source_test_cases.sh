#!/usr/bin/bash

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                  UQ4VLAs v1.0                    ┃
# ┃  Run uncertainty and quality experiments across  ┃
# ┃          models and datasets in style!           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
NC='\033[0m' # No color

# Usage message

show_help() {
  echo -e "${GREEN}SYNOPSIS${NC}"
  echo -e "  $0 -e <env> -m <model> [options]"
  echo -e ""
  echo -e "${GREEN}ARGUMENTS DESCRIPTION${NC}"
  echo -e "  ${YELLOW}-e${NC}  ${RED}[REQUIRED]${NC} Conda environment name to activate."
  echo -e "  ${YELLOW}-m${NC}  ${RED}[REQUIRED]${NC} Specific model name to run (e.g., eo1, gr00t)."
  echo -e "  ${YELLOW}-d${NC}  ${BLUE}[OPTIONAL]${NC} Filename of the dataset JSON."
  echo -e "      one or various separated with commas. Example: t-grasp_n-1000_o-m3_s-2498586606.json,t-move_n-1000_o-m3_s-2263834374.json"
  echo -e "      Default: Runs the 4 standard datasets (t-grasp_n-1000_o-m3_s-2498586606.json, t-move_n-1000_o-m3_s-2263834374.json, t-put-in_n-1000_o-m3_s-2905191776.json, t-put-on_n-1000_o-m3_s-2593734741.json)."
  echo -e "  ${YELLOW}-h${NC}  Show this help message."
  echo -e ""
  echo -e "${GREEN}USAGE EXAMPLE${NC}"
  echo -e "  $0 -e EO1 -m eo1 -d t-grasp_n-1000_o-m3_s-2498586606.json"
  exit 0
}

while getopts "e:m:d:h" opt; do
  case $opt in
    e) conda_env="$OPTARG" ;;
    m) specific_model="$OPTARG" ;;
    d) dataset="$OPTARG" ;;
    h) show_help ;;
    *) exit 1 ;;
  esac
done

CONDA_PATH=$(conda info --base)
source "$CONDA_PATH/etc/profile.d/conda.sh"
# Activate conda environment
echo -e "${BLUE}➤ Activating Conda environment: ${GREEN}${conda_env}${NC}"
conda activate "$conda_env" || {
  echo -e "${RED}✘ Failed to activate conda environment '${conda_env}'. Exiting.${NC}"
  exit 1
}

# Define datasets
datasets=(
  t-grasp_n-1000_o-m3_s-2498586606.json
  t-move_n-1000_o-m3_s-2263834374.json
  t-put-in_n-1000_o-m3_s-2905191776.json
  t-put-on_n-1000_o-m3_s-2593734741.json
)
if [[ -n "$dataset" ]]; then
  echo -e "${BLUE}➤ Running only for tasks: ${YELLOW}${dataset}${NC}"
  datasets_clean=$(echo $dataset | tr ',' ' ')
  
  # Convertimos el string en una matriz real
  datasets=($datasets_clean)
fi

# Start banner
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "      🚀 Launching Source Test cases"
echo -e "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Main experiment loop
for data in "${datasets[@]}"; do


  echo -e "\n${YELLOW}▶ Running model: ${specific_model} | Dataset: ${data}${NC}"
  MUJOCO_GL="glx" \
SAPIEN_RENDER_BACKEND="glx" \
xvfb-run -s "-screen 0 1600x1200x24 +extension GLX" -a python3.10 run_fuzzer_allMetrics.py -m "${specific_model}" -d "../data/${data}"

  if [[ $? -ne 0 ]]; then
    echo -e "${RED}✘ Failed: ${specific_model} on ${data}${NC}"
  else
    echo -e "${GREEN}✔ Completed: ${specific_model} on ${data}${NC}"
  fi

done

# Done!
echo -e "\n${GREEN}🎉 All experiments completed successfully.${NC}"
