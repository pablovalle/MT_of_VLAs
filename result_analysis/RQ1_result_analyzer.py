
import os
import json
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from frechetdist import frdist

LOW  = np.array([-1.0, -1.0, -1.0, -1.5707964, -1.5707964, -1.5707964, -1.0])
HIGH = np.array([ 1.0,  1.0,  1.0,  1.5707964,  1.5707964,  1.5707964,  1.0])

def getTrajectory(path):

    with open(path, "r") as f:
        data = json.load(f)

    trajectory = []

    raw = data * (HIGH - LOW) + LOW                      # (N, 7)

    # Extract only x,y,z
    pos = raw[:, :3]                                     # (N, 3)

    # Convert to list of tuples (if needed for frdist)
    return pos

    #return trajectory



def getVerdict(path):
    with open(path, 'r') as f:
        log_data = json.load(f)
        last_timestep_key = str(max(map(int, log_data.keys())))
        success_flag = log_data[last_timestep_key]["success"]
        label = 0 if success_flag=='false' or success_flag==False else 1
    return label

def calcualteOracleMR(verdict_orig, verdict_mt, mr):
    verdict=0
    if mr =="MR4":
        if verdict_orig!=verdict_mt:
            verdict=1
    else:
        if verdict_orig==verdict_mt:
            verdict=1

    return verdict
results = []

base_dir = "results"
model="eo1"
mt_results_dir="FollowUp_Results"
task_mapping={"grasp":"t-grasp_n-1000_o-m3_s-2498586606","move":"t-move_n-1000_o-m3_s-2263834374",
              "put-in":"t-put-in_n-1000_o-m3_s-2905191776","put-on":"t-put-on_n-1000_o-m3_s-2593734741"}

for model_res in os.listdir(mt_results_dir):
    if model_res==model:
        task_path = os.path.join(mt_results_dir, model_res)
        for mr in os.listdir(task_path):
            if mr!='MR5':
                continue
            for task in os.listdir(os.path.join(task_path, mr)):
                for task_id in os.listdir(os.path.join(task_path, mr, task)):
                    for follow_up_num in os.listdir(os.path.join(task_path, mr, task,task_id)):
                        #if task_id != "task_412" or mr != "MR3" or task != "put-in":
                        #    continue
                        
                        mt_curr_folder=os.path.join(task_path, mr, task,task_id,follow_up_num)
                        orig_folder=os.path.join(base_dir,task_mapping[task],model_res,"allMetrics", task_id.split("_")[-1])
                        print(f"Evaluating {mt_curr_folder}")
                        mt_tcp_path=os.path.join(mt_curr_folder,'tcp_poses.json')
                        orig_tcp_path=os.path.join(orig_folder,'tcp_poses.json')
                        traj_mt=getTrajectory(mt_tcp_path)
                        traj_orig=getTrajectory(orig_tcp_path)

                        frechet_distance = frdist(traj_mt, traj_orig)  # 1 is not violated, 0 is violated

                        mt_log_path=os.path.join(mt_curr_folder,'log.json')
                        orig_log_path=os.path.join(orig_folder,'log.json')

                        verdict_mt=getVerdict(mt_log_path)
                        verdict_orig=getVerdict(orig_log_path)
                        relation_verdict=calcualteOracleMR(verdict_orig, verdict_mt,mr)

                        results.append({
                            "model": model_res,
                            "task": task,
                            "scene_id": task_id.split("_")[-1],
                            "mr": mr,
                            "follow_up_num": follow_up_num.split("_")[-1],
                            "verdict_orig": verdict_orig,
                            "verdict_mt": verdict_mt,
                            "relation_verdict": relation_verdict,
                            "relation_distance": frechet_distance
                        })

df = pd.DataFrame(results)


# ---- Save to Excel ----
output_path = f"result_analysis/RQ1_results_{model}.xlsx"
df.to_excel(output_path, index=False)


df["task_mr"] = df["task"] + " | " + df["mr"]


# Palette for relation_verdict: 0=blue, 1=orange
verdict_palette = {0: "#4C72B0", 1: "#DD8452"}

sns.set_theme(style="whitegrid", font_scale=1.4)
plt.rcParams["axes.linewidth"] = 1.2
plt.figure(figsize=(18, 9))

# Boxplot: X = task+MR, Y = distance, hue = relation_verdict
ax = sns.boxplot(
    data=df,
    x="task_mr",
    y="relation_distance",
    hue="relation_verdict",
    palette=verdict_palette,
    showfliers=False,
    linewidth=1.1,
    dodge=True
)

# Add shadow effect to each box
for box in ax.artists:
    box.set_path_effects([pe.withSimplePatchShadow(offset=(1, -1), alpha=0.25)])
    box.set_linewidth(0.8)

# Labels & title
ax.set_title(
    f"Fréchet Distance by Task, MR, and Relation Verdict ({model})",
    fontsize=20,
    fontweight='bold',
    pad=22
)
ax.set_xlabel("Task | MR", fontsize=15)
ax.set_ylabel("Fréchet Distance", fontsize=15)

plt.xticks(rotation=45, ha="right", fontsize=12)
plt.yticks(fontsize=12)

# Legend outside
leg = ax.legend(
    title="Relation Verdict",
    title_fontsize=14,
    fontsize=12,
    frameon=True,
    facecolor="white",
    edgecolor="#dddddd",
    loc="upper left",
    bbox_to_anchor=(1.01, 1)
)
leg.get_frame().set_linewidth(1)

# --- Add vertical dashed lines where MR changes ---
# Get the x positions and task_mr labels
xticks_labels = df["task_mr"].unique()
# Extract MR part of the label
mrs_in_order = [label.split(" | ")[1] for label in xticks_labels]

# Draw vertical lines when MR changes
last_mr = mrs_in_order[0]
for i, mr in enumerate(mrs_in_order[1:], start=1):
    if mr != last_mr:
        ax.axvline(i - 0.5, color="gray", linestyle="--", linewidth=1)
        last_mr = mr

sns.despine(trim=True)
plt.tight_layout()

# Save figure
save_path = f"boxplot_{model}.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"Saved figure to: {save_path}")

plt.show()