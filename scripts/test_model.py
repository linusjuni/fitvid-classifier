"""Test any trained model on the test set."""
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

from src.datasets import FrameImageDataset
from src.models import ResNetBaseline
from src.training import test_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, default='resnet18', help='Model architecture')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = FrameImageDataset(split='test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Load model
    if args.model_type == 'resnet18':
        model = ResNetBaseline(num_classes=10, pretrained=False)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)
    
    # Test
    test_acc = test_model(model, test_loader, device)
    print(f"Test Accuracy: {test_acc:.2f}%")


if __name__ == '__main__':
    main()