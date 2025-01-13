import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Define overlap data as percentages (placeholder values; replace with actual computation results)
node_overlap_percent = np.array([
    [99.4, 60.3, 89.4, 82.7, 81.0, 61.0, 95.1],
    [60.3, 18.5, 67.4, 72.9, 74.3, 98.8, 63.3],
    [89.4, 67.4, 71.2, 92.4, 90.6, 68.2, 93.9],
    [82.7, 72.9, 92.4, 94.9, 98.0, 73.8, 86.8],
    [81.0, 74.3, 90.6, 98.0, 83.3, 75.2, 85.1],
    [61.0, 98.8, 68.2, 73.8, 75.2, 66.4, 64.1],
    [95.1, 63.3, 93.9, 86.8, 85.1, 64.1, 94.6]
])

edge_overlap_percent = np.array([
    [5.9, 5.3, 4.6, 5.3, 5.1, 4.1, 5.6],
    [5.3, 1.7, 3.7, 4.9, 4.8, 4.8, 4.6],
    [4.6, 3.7, 3.5, 4.1, 4.0, 3.0, 4.1],
    [5.3, 4.9, 4.1, 5.0, 4.9, 3.9, 4.9],
    [5.1, 4.8, 4.0, 4.9, 4.2, 3.6, 4.8],
    [4.1, 4.8, 3.0, 3.9, 3.6, 4.2, 3.4],
    [5.6, 4.6, 4.1, 4.9, 4.8, 3.4, 5.1]
])

labels = ["Benign", "XSS Stored", "XSS Reflected", "XSS DOM", "SQL Injection", 
          "Command Injection", "Brute Force"]

# Plot heatmaps
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(node_overlap_percent, annot=True, fmt=".1f", cmap="YlGnBu", cbar=True,
            xticklabels=labels, yticklabels=labels, ax=axes[0])
axes[0].set_title("Node Overlap (%)")
axes[0].set_xlabel("Graph Class")
axes[0].set_ylabel("Graph Class")

sns.heatmap(edge_overlap_percent, annot=True, fmt=".1f", cmap="YlOrRd", cbar=True,
            xticklabels=labels, yticklabels=labels, ax=axes[1])
axes[1].set_title("Edge Overlap (%)")
axes[1].set_xlabel("Graph Class")
axes[1].set_ylabel("Graph Class")

plt.tight_layout()
plt.show()
