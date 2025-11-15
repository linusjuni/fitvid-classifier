import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from pathlib import Path

from src.datasets import DualStreamDataset, FlowTransform
from src.models import TemporalStreamCNN
from src.training.train import train_temporal_epoch, validate_temporal
from src.utils.metrics import init_metrics, add_epoch_metrics, save_metrics
from src.utils.early_stopping import EarlyStopping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="no_leakage",
        choices=["leakage", "no_leakage"],
        help="Dataset to use: leakage or no_leakage (corrected)",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--min_delta", type=float, default=0.01, help="Minimum improvement for early stopping")
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze backbone and only train classifier")
    args = parser.parse_args()

    # Initialize metrics
    metrics = init_metrics()

    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta, mode='min')

    # Create checkpoints directory
    checkpoint_dir = Path(f"checkpoints/temporal_stream_{args.dataset}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Freeze backbone: {args.freeze_backbone}")

    # Data transforms
    # Note: We only need temporal transform, spatial is ignored
    spatial_transform = None  # Not used for temporal stream training
    temporal_transform = FlowTransform(size=(224, 224), normalize=True)

    # Datasets - using DualStreamDataset but only using flow data
    train_dataset = DualStreamDataset(
        dataset_name=args.dataset, 
        split="train", 
        spatial_transform=spatial_transform,
        temporal_transform=temporal_transform,
        frame_sample_strategy="random"  # Doesn't matter for temporal stream
    )
    val_dataset = DualStreamDataset(
        dataset_name=args.dataset, 
        split="val", 
        spatial_transform=spatial_transform,
        temporal_transform=temporal_transform,
        frame_sample_strategy="random"
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # Model - Temporal Stream
    model = TemporalStreamCNN(
        num_classes=10, 
        pretrained=True,  # Uses ImageNet weights (except first conv)
        freeze_backbone=args.freeze_backbone
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    best_val_acc = 0

    print("\nStarting temporal stream training...")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    for epoch in range(args.epochs):
        train_loss, train_acc = train_temporal_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate_temporal(model, val_loader, criterion, device)

        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        metrics = add_epoch_metrics(metrics, epoch + 1, train_loss, train_acc, val_loss, val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pth")
            print(f"  → Saved best model (val_acc: {val_acc:.2f}%)")

        # Check early stopping
        if early_stopping(val_loss):
            print(f"Early stopping triggered after epoch {epoch + 1}")
            break

    # Save final model
    torch.save(model.state_dict(), checkpoint_dir / "final_model.pth")
    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.2f}%")
    print(f"Models saved to {checkpoint_dir}")

    # Save metrics
    save_metrics(metrics, checkpoint_dir / "training_metrics.csv")


if __name__ == "__main__":
    main()
