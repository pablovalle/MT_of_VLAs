import os
import json
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib
import numpy as np
from matplotlib_venn import venn2
from matplotlib.lines import Line2D



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

base_dir = "results"
#model="openvla-7b"
mt_results_dir="FollowUp_Results"
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
        if threshold==0.3: threshold=0.1 
        elif threshold==0.1: threshold=0.3
        if row["relation_distance"]< threshold: #or row["relation_verdict"]> 2*threshold:
            return 0
        else:
            return 1

def detect_recall(row):
    if row["relation_verdict"] == 0 and row["dist_mr"]==0:
        return 1 # Are agree
    else:
        return 0 # Are disagree

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
        data=pd.read_excel(f"result_analysis/RQ2_results_{model_res}.xlsx") 
        array_of_dfs.append(data) 
    df=pd.concat(array_of_dfs) 
    print(len(df))

    # Compute detection
    df["dist_mr"] = df.apply(lambda row: detect_dist(row, threshold), axis=1)
    df["recall"] = df.apply(lambda row: detect_recall(row), axis=1)
    #print(df["recall"])
    # Aggregate
    total_counts = df.groupby(["model", "mr", "task"]).size().reset_index(name="total_count")
    zero_counts = df[df["dist_mr"] == 0].groupby(["model", "mr", "task"]).size().reset_index(name="zero_count")
    oracle_counts = df[df["relation_verdict"] == 0].groupby(["model", "mr", "task"]).size().reset_index(name="oracle_count")
    precision_counts= df[df["recall"]==1].groupby(["model", "mr", "task"]).size().reset_index(name="precision_counts")
    summary = total_counts.merge(zero_counts, on=["model", "mr", "task"], how="left")
    summary = summary.merge(precision_counts, on=["model", "mr", "task"], how="left")
    summary = summary.merge(oracle_counts, on=["model", "mr", "task"], how="left")
    summary["zero_count"] = summary["zero_count"].fillna(0)
    summary["precision_counts"] = summary["precision_counts"].fillna(0)
    summary["oracle_count"] = summary["oracle_count"].fillna(0)
    summary["zero_percentage"] = summary["zero_count"] / summary["total_count"] * 100
    summary["precision_percentage"] = summary["precision_counts"] / summary["zero_count"]    
    summary["recall_percentage"] = summary["precision_counts"] / summary["oracle_count"]    
    #print(recall_counts)
    # Add threshold column
    summary["Threshold"] = threshname

    all_rows.append(summary)
    output_cols = ["model", "task", "scene_id", "mr", "follow_up_num"]

    # 3. Filter for Case A: Oracle=1, Precision=0, MT=0
    group_a = df[
        (df["dist_mr"] == 1) & 
        (df["recall"]==0) & 
        (df["relation_verdict"] == 0)
    ]

    # Merge back with original df to get scene_id and follow_up_num

    result_a = group_a[output_cols]

    # Save Case A
    result_a.to_excel(f"case_oracle_only_{threshname}.xlsx", index=False)
    print(f"Saved Case A with {len(result_a)} rows.")
    print(len(result_a))
    # 4. Filter for Case B: Oracle=1, Precision=0, Zero=0
    group_b = df[
        (df["dist_mr"] == 0) & 
        (df["recall"]==0) & 
        (df["relation_verdict"] == 1)
    ]

    # Merge back with original df
    result_b = group_b[output_cols]

    # Save Case B
    result_b.to_excel(f"case_mr_only_{threshname}.xlsx", index=False)
    print(f"Saved Case B with {len(result_b)} rows.")
    print(len(result_b))

# Concatenate all thresholds
full_summary = pd.concat(all_rows, ignore_index=True)

# ---- Ordering ----
full_summary["task"] = pd.Categorical(full_summary["task"], categories=task_order, ordered=True)
full_summary["mr"] = pd.Categorical(full_summary["mr"], categories=mr_order, ordered=True)
full_summary["Threshold"] = pd.Categorical(full_summary["Threshold"], categories=["High", "Medium", "Low"], ordered=True)
model_order = sorted(full_summary["model"].unique())

threshold_summary = full_summary.groupby("Threshold")[["oracle_count", "zero_count", "precision_counts"]].sum()

# Reset index to make it a clean table
threshold_summary = threshold_summary.reset_index()

# Display the result
print(threshold_summary)
print(len(df))



row_index = []
rows = []

for model in model_order:
    for mr in mr_order:

        row = []
        for thresh in thresholds:
            if thresh==0.1:
                threshname="High"
            elif thresh==0.2:
                threshname="Medium"
            elif thresh==0.3:
                threshname="Low"
            for task in task_order:
                for metric in ["recall_percentage", "precision_percentage"]:
                    cell = full_summary[
                        (full_summary["model"] == model) &
                        (full_summary["mr"] == mr) &
                        (full_summary["Threshold"] == threshname) &
                        (full_summary["task"] == task)
                    ]
                    #print(cell)
                    if len(cell):
                        value = cell[metric].values[0]
                        value = 0.0 if pd.isna(value) else value
                    else:
                        value = 0.0

                    row.append(value)

        rows.append(row)
        row_index.append((model, mr))

# -----------------------------
# MultiIndex columns: Threshold → Task → Metric
# -----------------------------
columns = []
for thresh in thresholds:
    for task in task_order:
        columns.append((thresh, task, "Recall (%)"))
        columns.append((thresh, task, "Precision (%)"))

multi_cols = pd.MultiIndex.from_tuples(columns, names=["Threshold", "Task", "Metric"])

# -----------------------------
# Build DataFrame
# -----------------------------
matrix_df = pd.DataFrame(
    rows,
    index=pd.MultiIndex.from_tuples(row_index, names=["Model", "MR"]),
    columns=multi_cols
)

cmap = plt.cm.get_cmap("YlGnBu")
norm = matplotlib.colors.Normalize(vmin=0, vmax=1)  # Values are 0-100%

def value_to_rgb(val):
    rgba = cmap(norm(val))
    r, g, b, _ = rgba
    return r, g, b  # Matplotlib returns float 0-1

# -----------------------------
# Generate LaTeX table with colors
# -----------------------------
def df_to_latex_colored(df):
    latex_lines = []

    latex_lines.append("\\begin{tabular}{ll" + "c" * df.shape[1] + "}")
    latex_lines.append("\\toprule")

    # Header: flatten MultiIndex
    header_line1 = ["Model", "MR"]
    header_line2 = ["", ""]
    for thresh, task, metric in df.columns:
        header_line1.append(thresh)
        header_line2.append(f"{task}-{metric}")
    latex_lines.append(" & ".join([str(x) for x in header_line1]) + " \\\\")
    latex_lines.append(" & ".join([str(x) for x in header_line2]) + " \\\\")
    latex_lines.append("\\midrule")

    # Rows
    for idx, row in df.iterrows():
        line = [idx[0], idx[1]]
        for val in row:
            r, g, b = value_to_rgb(val)
            line.append(f"\\cellcolor[rgb]{{{r:.3f},{g:.3f},{b:.3f}}}{val:.1f}")
        latex_lines.append(" & ".join([str(x) for x in line]) + " \\\\")
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    return "\n".join(latex_lines)

latex_code = df_to_latex_colored(matrix_df)

with open("recall_precision_matrix_colored_heatmap.tex", "w") as f:
    f.write(latex_code)

print("LaTeX table saved as recall_precision_matrix_colored_heatmap.tex")
"""
# -----------------------------
# Export to LaTeX
# -----------------------------
latex_code = matrix_df.to_latex(
    float_format="%.2f",
    multicolumn=True,
    multirow=True,
    escape=False
)

with open("recall_precision_matrix_models_rows.tex", "w") as f:
    f.write(latex_code)

print("LaTeX table saved as recall_precision_matrix_models_rows.tex")
"""
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
                ]["precision_percentage"]
                row.append(float(v.values[0]) if not pd.isna(v.values[0]) else 0.0)
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
    fmt=".2f",
    cmap="YlGnBu",
    linewidths=0.7,
    linecolor="white",
    vmin=0,
    vmax=1,
    cbar_kws={"label": "Precision", "aspect": 40, "pad": 0.02},
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
plt.savefig("heatmap_precision_all.png", dpi=300, bbox_inches="tight")
plt.show()







threshold_names = ["High", "Medium", "Low"]
# model_order, mr_order, task_order should be defined in your environment

# 1. Grid Dimensions
# 1. Tighter Grid Dimensions
n_tasks = len(task_order)
n_thresh = len(threshold_names)
n_cols = n_thresh * n_tasks
n_rows = len(model_order) * len(mr_order)
mrs_per_model = len(mr_order)

# Adjust height: 1.4 per row is tight but readable
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 1.4))

if n_rows == 1: axes = axes.reshape(1, -1)
if n_cols == 1: axes = axes.reshape(-1, 1)



row_idx = 0
for m_idx, model in enumerate(model_order):
    for mr_idx, mr in enumerate(mr_order):
        col_idx = 0
        current_row_ax = axes[row_idx, 0]
        
        # --- Grouped Model Labels (Rotated & Centered) ---
        # Only draw the model name once per block of MRs
        if mr_idx == (mrs_per_model // 2):
            m_label = model_mapping.get(model, model)
            current_row_ax.text(-0.85, 0.5, m_label, transform=current_row_ax.transAxes,
                                fontsize=22, fontweight='bold', va='center', ha='center', 
                                rotation=90)

        for thresh in threshold_names:
            for task in task_order:
                ax = axes[row_idx, col_idx]
                
                cell_data = full_summary[
                    (full_summary["Threshold"] == thresh) &
                    (full_summary["task"] == task) &
                    (full_summary["model"] == model) &
                    (full_summary["mr"] == mr)
                ]

                if not cell_data.empty:
                    val_oracle = int(cell_data["oracle_count"].iloc[0])
                    val_metric = int(cell_data["zero_count"].iloc[0])
                    val_inter  = int(cell_data["precision_counts"].iloc[0])
                    
                    only_oracle = max(0, val_oracle - val_inter)
                    only_metric = max(0, val_metric - val_inter)
                    
                    v = venn2(subsets=(1, 1, 0.5), 
                              set_labels=('', ''), 
                              set_colors=('#FF6B6B', '#4D96FF'), 
                              alpha=0.6, ax=ax)
                    
                    if v:
                        labels = {'10': only_oracle, '01': only_metric, '11': val_inter}
                        for sid, val in labels.items():
                            label = v.get_label_by_id(sid)
                            if label:
                                label.set_text(str(val))
                                label.set_fontsize(16)
                                label.set_fontweight('bold')
                else:
                    ax.text(0.5, 0.5, "0", ha='center', va='center', fontsize=14)
                
                ax.axis('off')

                # Column Labels (Task) - Top row only
                if row_idx == 0:
                    ax.text(0.5, 1.1, f"{task}", transform=ax.transAxes,
                            fontsize=20, fontweight='bold', ha='center')
                
                # Row Labels (MR only) - First column only
                if col_idx == 0:
                    ax.text(-0.2, 0.5, mr, transform=ax.transAxes,
                            fontsize=20, fontweight='bold', va='center', ha='right')

                col_idx += 1
        row_idx += 1

# --- Global Grouped Headers (No padding issues) ---
# Adjusting margins to make room for rotated labels and top headers
plt.subplots_adjust(left=0.18, right=0.98, top=0.95, bottom=0.02, wspace=0.05, hspace=0.05)
for r in range(5, n_rows, 5):
    # Get the position of the axis above the split
    pos_above = axes[r-1, 0].get_position()
    pos_below = axes[r, 0].get_position()
    
    # Calculate the y-coordinate for the line (midway between rows)
    y_coord = (pos_above.y0 + pos_below.y1) / 2
    
    # Draw line from the MR label area to the end of the last column
    line = Line2D([0.12, 0.98], [y_coord, y_coord], 
                  transform=fig.transFigure, color='black', 
                  linestyle='--', linewidth=2.5, alpha=0.7)
    fig.add_artist(line)
for i, thresh in enumerate(threshold_names):
    # Calculate screen center for each 4-column group
    group_start = i * n_tasks
    group_end = (i + 1) * n_tasks - 1
    pos_left = axes[0, group_start].get_position()
    pos_right = axes[0, group_end].get_position()
    center_x = (pos_left.x0 + pos_right.x1) / 2
    
    # Threshold header sitting tightly above Task labels
    fig.text(center_x, 0.97, f"{thresh}", 
             ha='center', fontsize=25, fontweight='bold')

plt.savefig("tight_rotated_venn_matrix.png", dpi=300, bbox_inches="tight")
plt.show()