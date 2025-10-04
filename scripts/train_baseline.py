"""Train a simple per-frame ResNet18 baseline."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from pathlib import Path

from src.datasets import FrameImageDataset
from src.models import ResNetBaseline
from src.training import train_epoch, validate


def main():
    # Create checkpoints directory
    checkpoint_dir = Path('checkpoints/baseline_resnet18')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data transforms
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets
    train_dataset = FrameImageDataset(split='train', transform=transform)
    val_dataset = FrameImageDataset(split='val', transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Model
    model = ResNetBaseline(num_classes=10, pretrained=True).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Training loop
    num_epochs = 5
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_dir / 'best_model.pth')
            print(f"  → Saved best model (val_acc: {val_acc:.2f}%)")
    
    # Save final model
    torch.save(model.state_dict(), checkpoint_dir / 'final_model.pth')
    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.2f}%")
    print(f"Models saved to {checkpoint_dir}")


if __name__ == '__main__':
    main()