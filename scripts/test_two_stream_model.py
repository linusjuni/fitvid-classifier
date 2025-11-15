import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

from src.datasets import DualStreamDataset, FlowTransform
from src.models import SpatialStreamCNN, TemporalStreamCNN, TwoStreamNetwork


def test_two_stream(model, loader, device, test_individual=False):
    """Test two-stream network."""
    model.eval()
    correct = 0
    total = 0
    
    if test_individual:
        spatial_correct = 0
        temporal_correct = 0
    
    with torch.no_grad():
        for rgb_frames, flow_stack, labels in loader:
            rgb_frames = rgb_frames.to(device)
            flow_stack = flow_stack.to(device)
            labels = labels.to(device)
            
            # Two-stream prediction
            outputs = model(rgb_frames, flow_stack)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Individual stream predictions (for analysis)
            if test_individual:
                spatial_outputs = model.forward_spatial(rgb_frames)
                temporal_outputs = model.forward_temporal(flow_stack)
                
                _, spatial_pred = spatial_outputs.max(1)
                _, temporal_pred = temporal_outputs.max(1)
                
                spatial_correct += spatial_pred.eq(labels).sum().item()
                temporal_correct += temporal_pred.eq(labels).sum().item()
    
    accuracy = 100.0 * correct / total
    
    if test_individual:
        spatial_acc = 100.0 * spatial_correct / total
        temporal_acc = 100.0 * temporal_correct / total
        return accuracy, spatial_acc, temporal_acc
    
    return accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spatial_checkpoint', type=str, 
                        default='checkpoints/aggregation_2d_no_leakage/best_model.pth',
                        help='Path to spatial stream checkpoint')
    parser.add_argument('--temporal_checkpoint', type=str,
                        default='checkpoints/temporal_stream_no_leakage/best_model.pth',
                        help='Path to temporal stream checkpoint')
    parser.add_argument('--dataset', type=str, default='no_leakage',
                        choices=['leakage', 'no_leakage'])
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--fusion_weights', type=float, nargs=2, default=[0.5, 0.5],
                        help='Fusion weights [spatial, temporal]')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Fusion weights: spatial={args.fusion_weights[0]}, temporal={args.fusion_weights[1]}")
    
    # Load models
    print("\nLoading models...")
    
    # Spatial stream
    spatial_model = SpatialStreamCNN(num_classes=10, pretrained=False)
    spatial_checkpoint = torch.load(args.spatial_checkpoint, map_location=device)
    spatial_model.load_state_dict(spatial_checkpoint)
    print(f"Loaded spatial stream from: {args.spatial_checkpoint}")
    
    # Temporal stream
    temporal_model = TemporalStreamCNN(num_classes=10, pretrained=False)
    temporal_checkpoint = torch.load(args.temporal_checkpoint, map_location=device)
    temporal_model.load_state_dict(temporal_checkpoint)
    print(f"Loaded temporal stream from: {args.temporal_checkpoint}")
    
    # Create two-stream network
    two_stream = TwoStreamNetwork(
        num_classes=10,
        spatial_model=spatial_model,
        temporal_model=temporal_model,
        fusion_weights=tuple(args.fusion_weights)
    ).to(device)
    
    # Prepare test data
    spatial_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    temporal_transform = FlowTransform(size=(224, 224), normalize=True)
    
    test_dataset = DualStreamDataset(
        dataset_name=args.dataset,
        split='test',
        spatial_transform=spatial_transform,
        temporal_transform=temporal_transform,
        frame_sample_strategy='middle'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    print(f"\nTest samples: {len(test_dataset)}")
    
    # Test with individual stream analysis
    print("\n" + "="*60)
    print("Testing Two-Stream Network")
    print("="*60)
    
    two_stream_acc, spatial_acc, temporal_acc = test_two_stream(
        two_stream, test_loader, device, test_individual=True
    )
    
    print(f"\nResults:")
    print(f"  Spatial Stream Only:  {spatial_acc:.2f}%")
    print(f"  Temporal Stream Only: {temporal_acc:.2f}%")
    print(f"  Two-Stream Fusion:    {two_stream_acc:.2f}%")

if __name__ == '__main__':
    main()
