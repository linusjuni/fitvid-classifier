# test_model.py
"""Test any trained model on the test set."""
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

from src.datasets import FrameImageDataset, FrameVideoDataset
from src.models import ResNetBaseline
from src.training import test_model_single_frame, test_model_video_aggregation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, default='resnet18', help='Model architecture')
    parser.add_argument('--dataset', type=str, default='no_leakage', 
                        choices=['leakage', 'no_leakage'], help='Dataset to use')
    parser.add_argument('--test_mode', type=str, default='video', 
                        choices=['frame', 'video'], 
                        help='frame: per-frame accuracy, video: video-level with aggregation')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (for frame mode)')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Test mode: {args.test_mode}")
    
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load model
    if args.model_type == 'resnet18':
        model = ResNetBaseline(num_classes=10, pretrained=False)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    
    match args.test_mode:
        case 'frame':
            # Per-frame testing
            test_dataset = FrameImageDataset(
                dataset_name=args.dataset, split='test', transform=transform
            )
            test_loader = DataLoader(
                test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
            )
            test_acc = test_model_single_frame(model, test_loader, device)
            print(f"\nPer-Frame Test Accuracy: {test_acc:.2f}%")
            
        case 'video':
            # Video-level testing with aggregation
            test_dataset = FrameVideoDataset(
                dataset_name=args.dataset, split='test', transform=transform, stack_frames=True
            )
            test_loader = DataLoader(
                test_dataset, batch_size=1, shuffle=False, num_workers=4
            )
            test_acc = test_model_video_aggregation(model, test_loader, device)
            print(f"\nVideo-Level Test Accuracy (with aggregation): {test_acc:.2f}%")
            
        case _:
            raise ValueError(f"Unknown test mode: {args.test_mode}")


if __name__ == '__main__':
    main()