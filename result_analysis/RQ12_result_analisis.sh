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
  echo -e "  $0 -e <env>"
  echo -e ""
  echo -e "${GREEN}ARGUMENTS DESCRIPTION${NC}"
  echo -e "  ${YELLOW}-e${NC}  ${RED}[REQUIRED]${NC} Conda environment name to activate."
  echo -e "  ${YELLOW}-h${NC}  Show this help message."
  echo -e ""
  echo -e "${GREEN}USAGE EXAMPLE${NC}"
  echo -e "  $0 -e EO1"
  exit 0
}

while getopts "e:h" opt; do
  case $opt in
    e) conda_env="$OPTARG" ;;
    h) show_help ;;
    *) exit 1 ;;
  esac
done

if [[ -z "$conda_env" ]]; then
  echo -e "${RED}Error: Conda environment name (-e) is required.${NC}"
  echo -e "Usage: ${YELLOW}./RQ1_result_analisis.sh -e <env>${NC}"
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

echo -e "${BLUE}➤ Evaluating the results for all the models ${NC}"


# Start banner
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "      🚀 RQ1 result analisis"
echo -e "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Main experiment loop

echo -e "\n${GREEN}▶ Analizing Models${NC}"
  #python3.10 RQ1_result_analyzer.py

echo -e "\n${GREEN}▶ Analizing the MT thresholds${NC}"
python3.10 MT_threshold_estimation.py

echo -e "\n${GREEN}▶ Generating Venn Diagram (Figure 3)${NC}"
python3.10 RQ1_Venn.py

echo -e "\n${GREEN}▶ Generating Heatmap (Figure 4)${NC}"
python3.10 RQ1_hetamap_distances.py

# Done!
echo -e "\n${GREEN}🎉 Analisis for RQ1 and RQ2 finished.${NC}"
