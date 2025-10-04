"""Test script to verify dataset loading works correctly."""
import sys
sys.path.append('..')  # Add parent directory to path

from src.fitvid_classifier.datasets import FrameImageDataset, FrameVideoDataset
from torchvision import transforms as T

def test_frame_dataset():
    print("=" * 50)
    print("Testing FrameImageDataset")
    print("=" * 50)
    
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    
    dataset = FrameImageDataset(split='train', transform=transform)
    print(f"Dataset size: {len(dataset)}")
    
    # Load a few samples
    for i in range(3):
        frame, label = dataset[i]
        print(f"Sample {i}: Frame shape={frame.shape}, Label={label}")
    
    print()

def test_video_dataset():
    print("=" * 50)
    print("Testing FrameVideoDataset")
    print("=" * 50)
    
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    
    dataset = FrameVideoDataset(split='train', transform=transform, stack_frames=True)
    print(f"Dataset size: {len(dataset)}")
    
    # Load a sample video
    frames, label = dataset[0]
    print(f"Video frames shape: {frames.shape} (C, T, H, W)")
    print(f"Label: {label}")
    
    print()

if __name__ == '__main__':
    test_frame_dataset()
    test_video_dataset()
    print("✓ All tests passed!")