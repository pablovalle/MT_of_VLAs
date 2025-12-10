import os
import json
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import shutil

base_dir = "results_original"


TOTAL_EXECUTIONS = 500

summary = defaultdict(lambda: defaultdict(dict))

for task in os.listdir(base_dir):
    task_path = os.path.join(base_dir, task)
    if not os.path.isdir(task_path):
        continue

    for model in os.listdir(task_path):
        if model=="gr00t":
            model_path = os.path.join(task_path, model, 'allMetrics')

            success=0
            for scene in tqdm(os.listdir(model_path), desc=f"Calculating for model {model}"):
                log_path = os.path.join(model_path , scene, 'log.json')
        
                with open(log_path, 'r') as f:
                    log_data = json.load(f)
                    last_timestep_key = str(max(map(int, log_data.keys())))
                    success_flag = log_data[last_timestep_key]["success"]
                    label = 0 if success_flag=='false' or success_flag==False else 1
                    if label==1:
                        print(log_path)
                        task_type=task_path.split("/")[-1]
                        dst_dir = f"filtered_videos_gr00t/{task_type}"
                        dst = os.path.join(dst_dir, f"{scene}_simulation.mp4")

                        os.makedirs(dst_dir, exist_ok=True)  # Create destination directory if it doesn't exist
                        shutil.copy2(os.path.join(model_path , scene, f"{scene}_simulation.mp4"), dst)
                    success=success+label
        
