import matplotlib.pyplot as plt
import numpy as np
# Data
sample_size = np.array([1000, 10000, 50000, 100000, 200000, 300000, 400000, 500000])
absolute_values = np.array([115.00, 883.00, 3218.00, 4578.00, 7300.00, 9423.00, 11274.00, 12886.00])
percentage = np.array([11.50, 8.83, 6.44, 4.58, 3.65, 3.14, 2.82, 2.58])
'''

'''
# Plot with dual Y-axes
fig, ax1 = plt.subplots(figsize=(10,7))

# Left axis (absolute values)
ax1.set_xlabel("Sample Size", fontsize=12)
ax1.set_ylabel("Count of Unique sp2 Structures", fontsize=12, color="#517E84")
ax1.plot(sample_size, absolute_values, marker="o", color="#517E84", 
         linestyle="-", linewidth=1.8, markersize=6, label="Total Count")
ax1.tick_params(axis="y", colors="#517E84")

# Right axis (percentage)
ax2 = ax1.twinx()
ax2.set_ylabel("Percentage (%)", fontsize=12, color="#0B4F6C")
ax2.plot(sample_size, percentage, marker="s", color="#0B4F6C", 
         linestyle="--", linewidth=1.8, markersize=6, label="Percentage")
ax2.tick_params(axis="y", colors="#0B4F6C")
ax2.set_ylim(bottom=0)  # Set minimum value to zero for percentage axis

# Add point labels for percentage values
for i, (x, y) in enumerate(zip(sample_size, percentage)):
    ax2.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", 
                xytext=(0,10), ha='left', va='top', fontsize=9, color="#0B4F6C")

# Grid and ticks
ax1.grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.7)
ax1.set_xticks(sample_size)
ax1.set_xticklabels([f"{x//1000}k" for x in sample_size], rotation=75, ha='center')

# Title
fig.suptitle("Count of Unique sp2 Structures over Sample Size", fontsize=13)

# Legends (combined from both axes)
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="center right")

plt.tight_layout()
plt.savefig("convergence_plot.png", dpi=300)