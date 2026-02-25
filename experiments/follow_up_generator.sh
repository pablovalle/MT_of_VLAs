#!/usr/bin/bash

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                  MT4VLAs v1.0                    ┃
# ┃  Run uncertainty and quality experiments across  ┃
# ┃          models and datasets in style!           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
NC='\033[0m' # No color

show_help() {
  echo -e "${GREEN}SYNOPSIS${NC}"
  echo -e "  $0 -e <env> -m <model> [options]"
  echo -e ""
  echo -e "${GREEN}ARGUMENTS DESCRIPTION${NC}"
  echo -e "  ${YELLOW}-e${NC}  ${RED}[REQUIRED]${NC} Conda environment name to activate."
  echo -e "  ${YELLOW}-m${NC}  ${RED}[REQUIRED]${NC} Specific model name to run (e.g., gpt-4, llama-3)."
  echo -e "  ${YELLOW}-r${NC}  ${BLUE}[OPTIONAL]${NC} Metamorphic Relations (MRs). Accepts single or multiple values"
  echo -e "      separated by commas or spaces. Example: \"MR1,MR2,MR5\"."
  echo -e "      Default: MR1, MR2, MR3, MR4, MR5."
  echo -e "  ${YELLOW}-t${NC}  ${BLUE}[OPTIONAL]${NC} Specific Task IDs to filter the execution."
  echo -e "      in an array way separated with commas. Example: [1] or [1,2,3,4]"
  echo -e "      in a range of values separated with -. Example: [1-10] or [1-10,20-25]"
  echo -e "      Default: None (runs all tasks that have been marked as pass in the human evaluation)."
  echo -e "  ${YELLOW}-d${NC}  ${BLUE}[OPTIONAL]${NC} Filename of the dataset JSON."
  echo -e "      one or various separated with commas. Example: t-grasp_n-1000_o-m3_s-2498586606.json,t-move_n-1000_o-m3_s-2263834374.json"
  echo -e "      Default: Runs the 4 standard datasets (t-grasp_n-1000_o-m3_s-2498586606.json, t-move_n-1000_o-m3_s-2263834374.json, t-put-in_n-1000_o-m3_s-2905191776.json, t-put-on_n-1000_o-m3_s-2593734741.json)."
  echo -e "  ${YELLOW}-h${NC}  Show this help message."
  echo -e ""
  echo -e "${GREEN}USAGE EXAMPLE${NC}"
  echo -e "  $0 -e EO1 -m eo1 -r \"MR1,MR3\" -t [1-10,15,18] -d t-grasp_n-1000_o-m3_s-2498586606.json"
  exit 0
}

while getopts "e:m:r:t:d:h" opt; do
  case $opt in
    e) conda_env="$OPTARG" ;;
    m) specific_model="$OPTARG" ;;
    r) mr_one="$OPTARG" ;;
    t) task_ids="$OPTARG" ;;
    d) dataset="$OPTARG" ;;
    h) show_help ;;
    *) exit 1 ;;
  esac
done

if [[ -z "$conda_env" || -z "$specific_model" ]]; then
  echo -e "${RED}Error: Conda environment name (-e) and model (-m) are required.${NC}"
  echo -e "Usage: ${YELLOW}./run_Follow_up_generator.sh -e <env> -m <model> [-r <mr>] [-t <tasks>] [-d <dataset>]${NC}"
  exit 1
fi

# Conda init (adjust this if needed)
CONDA_PATH=$(conda info --base)
source "$CONDA_PATH/etc/profile.d/conda.sh"

# Activate conda environment
echo -e "${BLUE}➤ Activating Conda environment: ${GREEN}${conda_env}${NC}"
conda activate "$conda_env" || {
  echo -e "${RED}✘ Failed to activate conda environment '${conda_env}'. Exiting.${NC}"
  exit 1
}



# Define model list
models=(openvla-7b spatialvla-4b pi0 gr00t eo1)
if [[ -n "$specific_model" ]]; then
  echo -e "${BLUE}➤ Running only for model: ${YELLOW}${specific_model}${NC}"
  models=("$specific_model")
fi

mrs=(
  MR1
  MR2
  #MR3
  MR4
  MR5
)

if [[ -n "$mr_one" ]]; then
  echo -e "${BLUE}➤ Running only for MR: ${YELLOW}${mr_one}${NC}"
  mr_clean=$(echo $mr_one | tr ',' ' ')
  
  # Convertimos el string en una matriz real
  mrs=($mr_clean)
fi

tasks="None"
if [[ -n "$task_ids" ]]; then
  echo -e "${BLUE}➤ Running only for ids: ${YELLOW}${task_ids}${NC}"
  tasks="$task_ids"
fi

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
echo -e "      🚀 Launching Follow-up Test Generator"
echo -e "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Main experiment loop
for data in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    for mr in "${mrs[@]}"; do
      echo -e "\n${YELLOW}▶ Running: MR: ${mr} | Model: ${model} | Dataset: ${data}${NC}"
      python3.10 Follow_up_test_case_generator.py --mr "${mr}" --model "${model}" --dataset "../data/${data}" --tasks "${tasks}"

      if [[ $? -ne 0 ]]; then
        echo -e "${RED}✘ Failed: ${model} on ${data}${NC}"
      else
        echo -e "${GREEN}✔ Completed: ${model} on ${data}${NC}"
      fi
    done
  done
done

# Done!
echo -e "\n${GREEN}🎉 All experiments completed successfully.${NC}"
