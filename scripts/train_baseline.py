"""Train a simple per-frame ResNet18 baseline."""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from pathlib import Path

from src.datasets import FrameImageDataset
from src.models import ResNetBaseline
from src.training import train_epoch, validate
from src.utils.metrics import init_metrics, add_epoch_metrics, save_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="no_leakage",
        choices=["leakage", "no_leakage"],
        help="Dataset to use: leakage or no_leakage (corrected)",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    # Initialize metrics
    metrics = init_metrics()

    # Create checkpoints directory
    checkpoint_dir = Path(f"checkpoints/baseline_resnet18_{args.dataset}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Dataset: {args.dataset}")

    # Data transforms
    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Datasets
    train_dataset = FrameImageDataset(
        dataset_name=args.dataset, split="train", transform=transform
    )
    val_dataset = FrameImageDataset(
        dataset_name=args.dataset, split="val", transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # Model
    model = ResNetBaseline(num_classes=10, pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    best_val_acc = 0

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        metrics = add_epoch_metrics(metrics, epoch + 1, train_loss, train_acc, val_loss, val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pth")
            print(f"  → Saved best model (val_acc: {val_acc:.2f}%)")

    # Save final model
    torch.save(model.state_dict(), checkpoint_dir / "final_model.pth")
    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.2f}%")
    print(f"Models saved to {checkpoint_dir}")

    # Save metrics
    save_metrics(metrics, checkpoint_dir / "training_metrics.csv")


if __name__ == "__main__":
    main()
