import os
import json
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np



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

base_dir = "results_original"
model="openvla-7b"
mt_results_dir="FollowUp_Results"
task_mapping={"grasp":"t-grasp_n-1000_o-m3_s-2498586606","move":"t-move_n-1000_o-m3_s-2263834374",
              "put-in":"t-put-in_n-1000_o-m3_s-2905191776","put-on":"t-put-on_n-1000_o-m3_s-2593734741"}

for model_res in os.listdir(mt_results_dir):
    if model_res==model:
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

df = pd.DataFrame(results)

# ---- Save to Excel ----
output_path = f"result_analysis/RQ1_results_{model}.xlsx"
df.to_excel(output_path, index=False)

print(f"Saved results to {output_path}")

df_zero = df[df["relation_verdict"] == 0]

# Contamos total por (mr, task)
total_counts = df.groupby(["mr", "task"]).size().reset_index(name="total_count")

# Contamos cuántos tienen relation_verdict == 0 por (mr, task)
zero_counts = df_zero.groupby(["mr", "task"]).size().reset_index(name="zero_count")

# Unimos las tablas
summary = total_counts.merge(zero_counts, on=["mr", "task"], how="left")

# Rellenamos posibles NaN (casos sin ningún 0)
summary["zero_count"] = summary["zero_count"].fillna(0).astype(int)

# Calculamos porcentaje
summary["zero_percentage"] = summary["zero_count"] / summary["total_count"] * 100

# Ordenar por mr y task (opcional)
summary = summary.sort_values(["mr", "task"])

print(summary)


palette = sns.color_palette("Spectral", n_colors=summary["mr"].nunique())

sns.set_theme(style="whitegrid", font_scale=1.4)
plt.rcParams["axes.linewidth"] = 1.2

plt.figure(figsize=(16, 9))

ax = sns.barplot(
    data=summary,
    x="task",
    y="zero_percentage",
    hue="mr",
    palette=palette,
    edgecolor="black",
    linewidth=1.1
)

# --- Rounded bars & subtle designer shadows ---
for bar in ax.patches:
    bar.set_path_effects([pe.withSimplePatchShadow(offset=(1, -1), alpha=0.25)])
    bar.set_linewidth(0.8)

# --- Add percentage labels only if > 0 ---
for p in ax.patches:
    height = p.get_height()
    if height > 0.001:   # Avoid showing 0.0%
        ax.annotate(
            f"{height:.1f}%",
            (p.get_x() + p.get_width()/2., height),
            ha="center",
            va="bottom",
            fontsize=11,
            color="#1a1a1a",
            path_effects=[pe.withStroke(linewidth=2, foreground="white")]
        )

# --- Extend Y-axis to 100% ---
ax.set_ylim(0, 100)

# --- Labels & title ---
ax.set_title(
    f"Percentage of Violated MRs for {model}",
    fontsize=20,
    fontweight='bold',
    pad=22
)

ax.set_xlabel("Task", fontsize=15)
ax.set_ylabel("Percentage (%)", fontsize=15)

plt.xticks(rotation=35, ha="right", fontsize=12)
plt.yticks(fontsize=12)

# --- Legend outside for cleanliness ---
leg = ax.legend(
    title="Mutation Rule",
    title_fontsize=14,
    fontsize=12,
    frameon=True,
    facecolor="white",
    edgecolor="#dddddd",
    loc="upper left",
    bbox_to_anchor=(1.01, 1)
)
leg.get_frame().set_linewidth(1)

sns.despine(trim=True)
plt.tight_layout()
save_path = f"plot_{model}.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"Saved figure to: {save_path}")
plt.show()
