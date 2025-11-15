import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
sns.set_palette("muted")

# Automatically discover all metric files
checkpoint_path = Path("checkpoints")
metric_files = list(checkpoint_path.glob("*/training_metrics.csv"))

# Extract model names and splits from paths
model_split_pairs = []
for file_path in metric_files:
    # e.g., "aggregation_2d_leakage" -> split into model and dataset
    folder_name = file_path.parent.name
    
    # Split by the last occurrence of _leakage or _no_leakage
    # Check for _no_leakage first since it's longer and contains _leakage
    if folder_name.endswith("_no_leakage"):
        model_name = folder_name[:-len("_no_leakage")]
        split = "no_leakage"
    elif folder_name.endswith("_leakage"):
        model_name = folder_name[:-len("_leakage")]
        split = "leakage"
    else:
        print(f"Warning: Could not parse folder name: {folder_name}")
        continue
    
    model_split_pairs.append((model_name, split, file_path))

# Get unique models and splits
models = sorted(list(set([m for m, s, f in model_split_pairs])))
splits = sorted(list(set([s for m, s, f in model_split_pairs])))

print(f"Found {len(model_split_pairs)} training runs:")
for model, split, _ in model_split_pairs:
    print(f"  - {model} ({split})")
print()

# Color palette for models
colors = sns.color_palette("muted", n_colors=len(models))
model_colors = dict(zip(models, colors))

def load_metrics(model_name, dataset):
    """Load training metrics CSV for a given model and dataset"""
    for m, s, file_path in model_split_pairs:
        if m == model_name and s == dataset:
            return pd.read_csv(file_path)
    print(f"Warning: Metrics not found for {model_name}_{dataset}")
    return None

def plot_split(split_name, output_dir="plots"):
    """Create accuracy and loss plots for a specific split"""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # === ACCURACY PLOT ===
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot each model
    for model_name in models:
        df = load_metrics(model_name, split_name)
        
        if df is None:
            continue
            
        color = model_colors[model_name]
        
        # Accuracy plot
        ax.plot(df['epoch'], df['train_acc'], 
                label=f"{model_name} (train)", 
                color=color, linestyle='--', linewidth=2)
        ax.plot(df['epoch'], df['val_acc'], 
                label=f"{model_name} (val)", 
                color=color, linestyle='-', linewidth=2)
    
    # Configure accuracy plot
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'Training vs Validation Accuracy - {split_name.replace("_", " ").title()}', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_path / f"accuracy_{split_name}.png", dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_path / f'accuracy_{split_name}.png'}")
    plt.close()
    
    # === LOSS PLOT ===
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot each model
    for model_name in models:
        df = load_metrics(model_name, split_name)
        
        if df is None:
            continue
            
        color = model_colors[model_name]
        
        # Loss plot
        ax.plot(df['epoch'], df['train_loss'], 
                label=f"{model_name} (train)", 
                color=color, linestyle='--', linewidth=2)
        ax.plot(df['epoch'], df['val_loss'], 
                label=f"{model_name} (val)", 
                color=color, linestyle='-', linewidth=2)
    
    # Configure loss plot
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'Training vs Validation Loss - {split_name.replace("_", " ").title()}', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_path / f"loss_{split_name}.png", dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_path / f'loss_{split_name}.png'}")
    plt.close()

def main():
    """Generate all plots"""
    print("Generating training curve plots...")
    
    for split in splits:
        print(f"\nProcessing {split} split...")
        plot_split(split)
    
    print("\nDone! All plots saved to 'plots/' directory")

if __name__ == "__main__":
    main()