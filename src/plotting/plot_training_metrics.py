# To run this script, use: 
# python -m src.plotting.plot_training_metrics --title "Your Very Own Title"

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

sns.set_style("whitegrid")
sns.set_palette("muted")

def plot_training_metrics(checkpoint_dir: str = "checkpoints", output_dir: str = "plots", title: str = None):
    """
    Plot training metrics for all models in the checkpoints directory.
    
    Args:
        checkpoint_dir: Path to checkpoints directory
        output_dir: Path to save plots
        title: Optional title prefix for plots
    """
    checkpoint_path = Path(checkpoint_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Find all training_metrics.csv files
    metric_files = list(checkpoint_path.glob("*/training_metrics.csv"))
    
    if not metric_files:
        print(f"No training_metrics.csv files found in {checkpoint_dir}")
        return
    
    # Load all data and find max epochs for consistent x-axis
    data = {}
    max_epochs = 0
    for metric_file in metric_files:
        model_name = metric_file.parent.name
        df = pd.read_csv(metric_file)
        data[model_name] = df
        max_epochs = max(max_epochs, df['epoch'].max())
    
    # Add padding to x-axis
    x_limit = max_epochs + 5
    
    # Plot 1: Training Loss
    fig, ax = plt.subplots(figsize=(10, 6))
    for model_name, df in data.items():
        ax.plot(df['epoch'], df['train_loss'], marker='o', label=model_name, linewidth=2, markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    plot_title = f'{title} - Training Loss' if title else 'Training Loss'
    ax.set_title(plot_title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, x_limit)
    plt.tight_layout()
    output_file = output_path / 'training_loss.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.close()
    
    # Plot 2: Validation Loss
    fig, ax = plt.subplots(figsize=(10, 6))
    for model_name, df in data.items():
        ax.plot(df['epoch'], df['val_loss'], marker='o', label=model_name, linewidth=2, markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    plot_title = f'{title} - Validation Loss' if title else 'Validation Loss'
    ax.set_title(plot_title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, x_limit)
    plt.tight_layout()
    output_file = output_path / 'validation_loss.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.close()
    
    # Plot 3: Training Accuracy
    fig, ax = plt.subplots(figsize=(10, 6))
    for model_name, df in data.items():
        ax.plot(df['epoch'], df['train_acc'], marker='o', label=model_name, linewidth=2, markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    plot_title = f'{title} - Training Accuracy' if title else 'Training Accuracy'
    ax.set_title(plot_title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, x_limit)
    plt.tight_layout()
    output_file = output_path / 'training_accuracy.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.close()
    
    # Plot 4: Validation Accuracy
    fig, ax = plt.subplots(figsize=(10, 6))
    for model_name, df in data.items():
        ax.plot(df['epoch'], df['val_acc'], marker='o', label=model_name, linewidth=2, markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    plot_title = f'{title} - Validation Accuracy' if title else 'Validation Accuracy'
    ax.set_title(plot_title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, x_limit)
    plt.tight_layout()
    output_file = output_path / 'validation_accuracy.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.close()
    
    print(f"\nAll plots saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot training metrics from checkpoint directories')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', 
                        help='Path to checkpoints directory (default: checkpoints)')
    parser.add_argument('--output-dir', type=str, default='plots',
                        help='Path to save plots (default: plots)')
    parser.add_argument('--title', type=str, default=None,
                        help='Optional title prefix for plots')
    
    args = parser.parse_args()
    
    plot_training_metrics(
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        title=args.title
    )