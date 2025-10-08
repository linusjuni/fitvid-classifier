"""Utilities for storing and managing training metrics."""
import polars as pl
from pathlib import Path


def init_metrics():
    """Initialize an empty metrics tracker."""
    return []


def add_epoch_metrics(metrics, epoch, train_loss, train_acc, val_loss, val_acc):
    """Add metrics for a single epoch."""
    metrics.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
    })
    return metrics


def save_metrics(metrics, save_path):
    """Save metrics to CSV file using polars."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    df = pl.DataFrame(metrics)
    df.write_csv(save_path)
    print(f"Metrics saved to {save_path}")


def load_metrics(csv_path):
    """Load metrics from CSV file."""
    return pl.read_csv(csv_path)