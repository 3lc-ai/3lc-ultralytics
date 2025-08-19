updated = [
    313031168,
    150629888,
    313031168,
    150629888,
    313031168,
    150629888,
    313031168,
    150629888,
    313031168,
    150629888,
    313031168,
    150629888,
]

develop = [
    566839040,
    490769088,
    906978240,
    830908288,
    1247117440,
    1171047488,
    1587256640,
    1511186688,
    1927395840,
    1851325888,
    2267535040,
    2191465088,
]

import matplotlib.pyplot as plt
import numpy as np

# Convert bytes to MB for better readability
updated_mb = [x / 1024 / 1024 for x in updated]
develop_mb = [x / 1024 / 1024 for x in develop]

# X-axis represents number of tables written
x_axis = range(1, len(updated) + 1)

# Create the plot with a modern style
plt.style.use("seaborn-v0_8" if "seaborn-v0_8" in plt.style.available else "default")
fig, ax = plt.subplots(figsize=(12, 8))

# Plot both lines with distinct colors and styles
ax.plot(
    x_axis,
    updated_mb,
    marker="o",
    linewidth=2.5,
    markersize=8,
    color="#2E8B57",
    label="Updated Version",
    markerfacecolor="white",
    markeredgewidth=2,
)

ax.plot(
    x_axis,
    develop_mb,
    marker="s",
    linewidth=2.5,
    markersize=8,
    color="#DC143C",
    label="Develop Version",
    markerfacecolor="white",
    markeredgewidth=2,
)

# Customize the plot
ax.set_xlabel("Number of Tables Written", fontsize=14, fontweight="bold")
ax.set_ylabel("PyArrow Memory Usage (MB)", fontsize=14, fontweight="bold")
ax.set_title("PyArrow Memory Usage Comparison\nUpdated vs Develop Version", fontsize=16, fontweight="bold", pad=20)

# Add grid for better readability
ax.grid(True, alpha=0.3, linestyle="--")

# Customize legend
ax.legend(loc="upper left", fontsize=12, frameon=True, fancybox=True, shadow=True, framealpha=0.9)

# Set axis limits and ticks
ax.set_xlim(0.5, len(updated) + 0.5)
ax.set_xticks(x_axis)
ax.set_ylim(0, max(develop_mb) * 1.1)

# Add some styling
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(1.5)
ax.spines["bottom"].set_linewidth(1.5)

# Add value annotations for key points
for i in [0, len(updated) - 1]:  # First and last points
    ax.annotate(
        f"{updated_mb[i]:.1f} MB",
        xy=(i + 1, updated_mb[i]),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=10,
        ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#2E8B57", alpha=0.7, edgecolor="none"),
        color="white",
        fontweight="bold",
    )

    ax.annotate(
        f"{develop_mb[i]:.1f} MB",
        xy=(i + 1, develop_mb[i]),
        xytext=(10, -20),
        textcoords="offset points",
        fontsize=10,
        ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#DC143C", alpha=0.7, edgecolor="none"),
        color="white",
        fontweight="bold",
    )

# Tight layout and display
plt.tight_layout()
plt.show()

# Optional: Save the plot
plt.savefig("memory_comparison.png", dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
print("Plot saved as 'memory_comparison.png'")
