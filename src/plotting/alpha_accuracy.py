import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set seaborn style
sns.set_style("whitegrid")
sns.set_palette("muted")

# Data from your experiments
alphas = [0.3, 0.4, 0.5, 0.6, 0.7]
spatial_acc = 50.83
temporal_acc = 63.33
fusion_acc = [75.83, 72.50, 70.00, 63.33, 58.33]

# Get muted palette colors
colors = sns.color_palette("muted", 3)

# Set up the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Set width of bars and positions
x = np.arange(len(alphas))
width = 0.5

# Create bars for fusion
bars = ax.bar(x, fusion_acc, width, label='Two-Stream Fusion', 
              color=colors[2], alpha=0.9)

# Add horizontal dashed lines for baselines
ax.axhline(y=spatial_acc, color=colors[0], linestyle='--', linewidth=5, 
           label=f'Spatial Stream ({spatial_acc:.2f}%)', alpha=0.8)
ax.axhline(y=temporal_acc, color=colors[1], linestyle='--', linewidth=5, 
           label=f'Temporal Stream ({temporal_acc:.2f}%)', alpha=0.8)

# Customize the plot
ax.set_xlabel(r'Fusion Weight $\alpha$ (Spatial Weight)', fontsize=12, fontweight='bold')
ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Two-Stream Network: Effect of Fusion Weight', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'{a:.1f}' for a in alphas])
ax.legend(fontsize=11, frameon=True)
ax.set_ylim([45, 80])

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('plots/fusion_weights_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('plots/fusion_weights_comparison.pdf', bbox_inches='tight')
print("Plot saved to plots/fusion_weights_comparison.png and .pdf")