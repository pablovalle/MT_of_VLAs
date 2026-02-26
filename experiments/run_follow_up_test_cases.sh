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
  echo -e "  ${YELLOW}-d${NC}  ${BLUE}[OPTIONAL]${NC} Filename of the dataset JSON."
  echo -e "      one or various separated with commas. Example: grasp,move"
  echo -e "      Default: Runs the 4 standard datasets (grasp, move, put-in, put-on)."
  echo -e "  ${YELLOW}-h${NC}  Show this help message."
  echo -e ""
  echo -e "${GREEN}USAGE EXAMPLE${NC}"
  echo -e "  $0 -e EO1 -m eo1 -r \"MR1,MR3\" -d grasp"
  exit 0
}

while getopts "e:m:r:d:h" opt; do
  case $opt in
    e) conda_env="$OPTARG" ;;
    m) specific_model="$OPTARG" ;;
    r) mr_one="$OPTARG" ;;
    d) dataset="$OPTARG" ;;
    h) show_help ;;
    *) exit 1 ;;
  esac
done

if [[ -z "$conda_env" || -z "$specific_model" ]]; then
  echo -e "${RED}Error: Conda environment name (-e) and model (-m) are required.${NC}"
  echo -e "Usage: ${YELLOW}./run_Follow_up_generator.sh -e <env> -m <model> [-r <mr>] [-d <dataset>]${NC}"
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

echo -e "${BLUE}➤ Running only for model: ${YELLOW}${specific_model}${NC}"


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
  grasp
  move
  put-on
  put-in
)
formatted_list=$(printf "'%s', " "${datasets[@]}")
# Remove the trailing comma and space, then wrap in brackets
python_tasks_array="[${formatted_list%, }]"


if [[ -n "$dataset" ]]; then
  echo -e "${BLUE}➤ Running only for tasks: ${YELLOW}${dataset}${NC}"
  datasets_clean=$(echo $dataset | tr ',' ' ')
  
  # Convertimos el string en una matriz real
  datasets=($datasets_clean)
fi



# Start banner
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "      🚀 Launching Follow-up Test Launcher"
echo -e "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Main experiment loop

for mr in "${mrs[@]}"; do
  echo -e "\n${YELLOW}▶ Running: MR: ${mr} | Model: ${specific_model} | Dataset: ${data}${NC}"
  MUJOCO_GL="glx" \
SAPIEN_RENDER_BACKEND="glx" \
xvfb-run -s "-screen 0 1600x1200x24 +extension GLX" -a python3.10 Follow_up_test_case_launcher.py --mr "${mr}" --model "${specific_model}" --tasks "${python_tasks_array}"

  if [[ $? -ne 0 ]]; then
    echo -e "${RED}✘ Failed: ${specific_model} on ${data}${NC}"
  else
    echo -e "${GREEN}✔ Completed: ${specific_model} on ${data}${NC}"
  fi
done


# Done!
echo -e "\n${GREEN}🎉 All experiments completed successfully.${NC}"
