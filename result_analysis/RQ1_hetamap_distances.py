import os
import json
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import math
OFFSET=math.sqrt(0.1**2+0.1**2)


def getVerdict(path):
    with open(path, 'r') as f:
        log_data = json.load(f)
        last_timestep_key = str(max(map(int, log_data.keys())))
        success_flag = log_data[last_timestep_key]["success"]
        label = 0 if success_flag=='false' or success_flag==False else 1
    return label

def calcualteOracleMR(verdict_orig, verdict_mt, mr):
    verdict=0
    if mr =="V_MR1":
        if verdict_orig!=verdict_mt:
            verdict=1
    else:
        if verdict_orig==verdict_mt:
            verdict=1

    return verdict

results = []

base_dir = "../results"
#model="openvla-7b"
mt_results_dir="../FollowUp_Results"
task_mapping={"grasp":"t-grasp_n-1000_o-m3_s-2498586606","move":"t-move_n-1000_o-m3_s-2263834374",
              "put-in":"t-put-in_n-1000_o-m3_s-2905191776","put-on":"t-put-on_n-1000_o-m3_s-2593734741"}
model_mapping={"eo1": "EO-1", "gr00t": "GR00T-N1.5", "openvla-7b": "OpenVLA-7B", "pi0": "pi0", "spatialvla-4b": "SpatialVLA-4B"}
task_order = ["grasp", "move", "put-in", "put-on"]
mr_order = ["MR1", "MR2", "MR3", "MR4", "MR5"]
"""
for model_res in os.listdir(mt_results_dir):
    #if model_res==model:
    print(f"Evaluating Model {model_res}")
    task_path = os.path.join(mt_results_dir, model_res)
    for mr in os.listdir(task_path):
        for task in os.listdir(os.path.join(task_path, mr)):
            for task_id in os.listdir(os.path.join(task_path, mr, task)):
                for follow_up_num in os.listdir(os.path.join(task_path, mr, task,task_id)):
                    mt_curr_folder=os.path.join(task_path, mr, task,task_id,follow_up_num)
                    orig_folder=os.path.join(base_dir,task_mapping[task],model_res,"allMetrics", task_id.split("_")[-1])

                    mt_log_path=os.path.join(mt_curr_folder,'log.json')
                    orig_log_path=os.path.join(orig_folder,'log.json')

                    verdict_mt=getVerdict(mt_log_path)
                    verdict_orig=getVerdict(orig_log_path)

                    relation_verdict=calcualteOracleMR(verdict_orig, verdict_mt,mr) # 1 is detected, 0 is not detected
                    results.append({
                        "model": model_res,
                        "task": task,
                        "scene_id": task_id.split("_")[-1],
                        "mr": mr,
                        "follow_up_num": follow_up_num.split("_")[-1],
                        "verdict_orig": verdict_orig,
                        "verdict_mt": verdict_mt,
                        "relation_verdict": relation_verdict
                    })
"""
def detect_dist(row, threshold):
    if threshold==0:
        return row["relation_verdict"]
    # example logic — replace with yours
    if row["mr"] in ["MR1", "MR2", "MR3"]:
        if row["relation_distance"]<threshold:
            return 1
        else:
            return 0
    if row["mr"] == "MR4":
        if threshold==0.3: threshold=0.1 
        elif threshold==0.1: threshold=0.3
        if row["relation_distance"]<threshold:
            return 0
        else:
            return 1
    if row["mr"] == "MR5":
        #if threshold==0.3: threshold=0.1 
        #elif threshold==0.1: threshold=0.3
        if row["relation_distance"]< OFFSET-threshold*0.5 or row["relation_distance"] > OFFSET+threshold*0.85:
            return 0
        else:
            return 1

thresholds=[0.1,0.2,0.3]

all_rows = []

for threshold in thresholds:
    if threshold ==0:
        threshname="Symbolic Oracle"
    if threshold==0.1:
        threshname="High"
    elif threshold==0.2:
        threshname="Medium"
    elif threshold==0.3:
        threshname="Low"

    # Load and concat data
    array_of_dfs = []
    for model_res in os.listdir(mt_results_dir): 
        data=pd.read_excel(f"RQ1_results_{model_res}.xlsx") 
        array_of_dfs.append(data) 
    df=pd.concat(array_of_dfs) 
    print(len(df))

    # Compute detection
    df["dist_mr"] = df.apply(lambda row: detect_dist(row, threshold), axis=1)

    # Aggregate
    total_counts = df.groupby(["model", "mr", "task"]).size().reset_index(name="total_count")
    zero_counts = df[df["dist_mr"] == 0].groupby(["model", "mr", "task"]).size().reset_index(name="zero_count")
    summary = total_counts.merge(zero_counts, on=["model", "mr", "task"], how="left")
    summary["zero_count"] = summary["zero_count"].fillna(0)
    summary["zero_percentage"] = summary["zero_count"] / summary["total_count"] * 100

    # Add threshold column
    summary["Threshold"] = threshname

    all_rows.append(summary)

# Concatenate all thresholds
full_summary = pd.concat(all_rows, ignore_index=True)

# ---- Ordering ----
full_summary["task"] = pd.Categorical(full_summary["task"], categories=task_order, ordered=True)
full_summary["mr"] = pd.Categorical(full_summary["mr"], categories=mr_order, ordered=True)
full_summary["Threshold"] = pd.Categorical(full_summary["Threshold"], categories=["High", "Medium", "Low"], ordered=True)
model_order = sorted(full_summary["model"].unique())

# ---- Build single heatmap DataFrame ----
pivot_data = []
y_labels = []

for thresh in ["High", "Medium", "Low"]:
    for task in task_order:
        row = []
        for model in model_order:
            for mr in mr_order:
                v = full_summary[
                    (full_summary["Threshold"] == thresh) &
                    (full_summary["task"] == task) &
                    (full_summary["model"] == model) &
                    (full_summary["mr"] == mr)
                ]["zero_percentage"]
                row.append(float(v.values[0]) if len(v) else 0.0)
        pivot_data.append(row)
        y_labels.append(f"{task}")

columns = [(m, mr) for m in model_order for mr in mr_order]
multi_cols = pd.MultiIndex.from_tuples(columns)

heatmap_df = pd.DataFrame(pivot_data, index=y_labels, columns=multi_cols)

# ---- Plot ----
plt.figure(figsize=(3 + 2.75 * len(mr_order), 2 * len(y_labels)/len(task_order)))
sns.heatmap(
    heatmap_df,
    annot=True,
    fmt=".1f",
    cmap="YlGnBu",
    linewidths=0.7,
    linecolor="white",
    vmin=0,
    vmax=100,
    cbar_kws={"label": "MR Violation Rate (%)", "aspect": 40, "pad": 0.02},
    #annot_kws={"fontweight": "bold"}
)

# Vertical separators between models
step = len(mr_order)
step_y = len(task_order)
for col in range(step, heatmap_df.shape[1], step):
    plt.vlines(col, *plt.gca().get_ylim(), colors="white", linewidth=5)

for row in range(step_y, heatmap_df.shape[0], step_y):
    plt.hlines(row, *plt.gca().get_xlim(), colors="white", linewidth=5)

# X labels for MR only
simple_labels = [c[1] for c in heatmap_df.columns]
plt.xticks(np.arange(len(simple_labels))+0.5, simple_labels, rotation=0, ha="center", fontsize=10)
plt.xlabel("")
# Model labels on top
for i, model in enumerate(model_order):

    plt.text(step*i + step/2, 13, model_mapping.get(model), ha="center", va="center", fontsize=10, fontweight="bold")
for i,threshold in enumerate(thresholds):
    if threshold==0:
        threshname="Symbolic Oracle"
    if threshold==0.1:
        threshname="High"
    elif threshold==0.2:
        threshname="Medium"
    elif threshold==0.3:
        threshname="Low"
    plt.text(-1.35 , step_y*i + step_y/2, threshname,rotation=90, ha="center", va="center", fontsize=10, fontweight="bold")

#plt.ylabel("Threshold - Task", fontsize=12)
plt.tight_layout()
plt.savefig("figures/heatmap_all_thresholds_single.png", dpi=300, bbox_inches="tight")
plt.show()