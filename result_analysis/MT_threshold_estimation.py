import os
import json
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import shutil
import matplotlib.pyplot as plt
import numpy as np

data_eo1=pd.read_excel("RQ1_results_eo1.xlsx")
data_gr00t=pd.read_excel("RQ1_results_gr00t.xlsx")
data_openvla=pd.read_excel("RQ1_results_openvla-7b.xlsx")
data_pi0=pd.read_excel("RQ1_results_pi0.xlsx")
data_spatialvla=pd.read_excel("RQ1_results_spatialvla-4b.xlsx")


all_data = pd.concat([data_eo1, data_gr00t, data_openvla, data_pi0, data_spatialvla], ignore_index=True)

# Assume your Excel has a column 'MR' identifying the relation (e.g., 'MR1', 'MR2', etc.)
# and the distance column is 'relation_distance'
# Filter for TC MRs: MR1 (Synonym), MR2 (Non-Interfering Object), MR3 (Light Change)
tc_data = all_data[all_data['mr'].isin(['MR1', 'MR2', 'MR3', 'MR4', 'MR5'])]

tc_distances = tc_data['relation_distance']

# Compute statistics
mean_dist = tc_distances.mean()
std_dist = tc_distances.std()
p20 = tc_distances.quantile(0.2)
p50 = tc_distances.quantile(0.50)
p80 =  tc_distances.quantile(0.80)
mean_plus_3std = mean_dist + 3 * std_dist
num_tc_pairs = len(tc_distances)

print(f"Number of TC pairs: {num_tc_pairs}")
print(f"Mean: {mean_dist:.4f} m")
print(f"Std: {std_dist:.4f} m")
print(f"20th percentile: {p20:.4f} m")
print(f"50th percentile: {p50:.4f} m")
print(f"80th percentile: {p80:.4f} m")

distances = tc_data['relation_distance'].dropna()

# Sort for CDF
sorted_distances = np.sort(distances)
n = len(sorted_distances)
cumulative_percentage = np.linspace(0, 100, n)

# Create the plot
plt.figure(figsize=(9, 6))
plt.plot(sorted_distances, cumulative_percentage, linewidth=2.5, color='#1f77b4', label='CDF of TC distances')

# Annotate your chosen thresholds
thresholds = {
    0.1: ('High strictness (0.1 m)', 'red'),
    0.2: ('Medium strictness (0.2 m)', 'orange'),
    0.3: ('Low strictness (0.3 m)', 'green')
}

for thresh, (label, color) in thresholds.items():
    # Find approximate cumulative % at threshold
    idx = np.searchsorted(sorted_distances, thresh)
    y_val = cumulative_percentage[idx - 1] if idx > 0 else 0
    if idx == n:
        y_val = 100
    plt.axvline(x=thresh, color=color, linestyle='--', linewidth=1.5, alpha=0.8)
    plt.text(thresh + 0.015, y_val + 4, label, color=color, fontsize=11, fontweight='bold')

# Annotate your reported percentiles
percentile_points = [
    (0.0941, "20th percentile", 20),
    (0.1916, "50th percentile (median)", 50),
    (0.3358, "80th percentile", 80)
]

for val, label, perc in percentile_points:
    idx = np.searchsorted(sorted_distances, val)
    y_val = cumulative_percentage[idx - 1] if idx > 0 else 0
    plt.plot(val, perc, 'o', color='purple', markersize=8)
    plt.text(val + 0.015, perc + 6, label, color='purple', fontsize=10, fontweight='semibold')

# Formatting
plt.title('Cumulative Distribution of Fréchet Distances\n'
          'for Trajectory Consistency Relations (MR1–MR3, N = 5,586 pairs)',
          fontsize=14, pad=20)
plt.xlabel('Fréchet Distance (meters)', fontsize=12)
plt.ylabel('Cumulative Percentage of Pairs (%)', fontsize=12)
plt.xlim(0, sorted_distances.max() * 1.05)
plt.ylim(0, 105)
plt.grid(True, alpha=0.3, linestyle=':')
plt.legend(loc='lower right', fontsize=10)
plt.tight_layout()

# Save high-quality figures for the paper
plt.savefig("figures/tc_frechet_cdf.pdf", dpi=300, bbox_inches='tight')

# Show plot
plt.show()