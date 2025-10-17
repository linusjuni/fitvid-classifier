import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

from src.datasets import FrameImageDataset, FrameVideoDataset
from src.models import AggregationModel2D, LateFusionModel2D, EarlyFusionModel2D, R3DModel
from src.training import test_model_single_frame, test_model_video_aggregation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['aggregation_2d', 'late_fusion_2d', 'early_fusion_2d', 'r3d'],
                        help='Model architecture type')
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
    print(f"Model type: {args.model_type}")
    print(f"Test mode: {args.test_mode}")
    
    # Different transforms for R3D vs other models
    if args.model_type == 'r3d':
        # R3D uses 112x112 and Kinetics normalization
        transform = T.Compose([
            T.Resize((112, 112)),
            T.ToTensor(),
            T.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
        ])
    else:
        # 2D models use 224x224 and ImageNet normalization
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # Load model based on type
    if args.model_type == 'aggregation_2d':
        model = AggregationModel2D(num_classes=10, pretrained=False)
    elif args.model_type == 'late_fusion_2d':
        model = LateFusionModel2D(num_classes=10, pretrained=False)
    elif args.model_type == 'early_fusion_2d':
        model = EarlyFusionModel2D(num_classes=10, num_frames=10, pretrained=False)
    elif args.model_type == 'r3d':
        model = R3DModel(num_classes=10, pretrained=False)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    # Load trained weights
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    print(f"Loaded model from: {args.model_path}")
    
    # Choose dataset and test mode based on model type and test_mode argument
    if args.model_type == 'aggregation_2d':
        # Aggregation model can do both frame and video testing
        if args.test_mode == 'frame':
            # Per-frame testing
            test_dataset = FrameImageDataset(
                dataset_name=args.dataset, split='test', transform=transform
            )
            test_loader = DataLoader(
                test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
            )
            test_acc = test_model_single_frame(model, test_loader, device)
            print(f"\nPer-Frame Test Accuracy: {test_acc:.2f}%")
            
        else:  # video mode
            # Video-level testing with aggregation
            test_dataset = FrameVideoDataset(
                dataset_name=args.dataset, split='test', transform=transform, stack_frames=True
            )
            test_loader = DataLoader(
                test_dataset, batch_size=1, shuffle=False, num_workers=4
            )
            test_acc = test_model_video_aggregation(model, test_loader, device)
            print(f"\nVideo-Level Test Accuracy (with aggregation): {test_acc:.2f}%")
    
    else:
        # Other models (late_fusion_2d, early_fusion_2d, r3d) only work with video data
        if args.test_mode == 'frame':
            print(f"Warning: Model type '{args.model_type}' requires video input. Switching to video mode.")
        
        test_dataset = FrameVideoDataset(
            dataset_name=args.dataset, split='test', transform=transform, stack_frames=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=4
        )
        
        # These models already process all frames, so we just do standard testing
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for frames, label in test_loader:
                frames, label = frames.to(device), label.to(device)
                outputs = model(frames)
                _, predicted = outputs.max(1)
                total += 1
                correct += (predicted == label).item()
        
        test_acc = 100. * correct / total
        print(f"\nVideo-Level Test Accuracy: {test_acc:.2f}%")


if __name__ == '__main__':
    main()