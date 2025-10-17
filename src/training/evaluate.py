"""Evaluation utilities."""
import torch


def test_model_single_frame(model, loader, device):
    """Test accuracy on individual frames (per-frame evaluation)."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for frames, labels in loader:
            frames, labels = frames.to(device), labels.to(device)
            outputs = model(frames)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy


def test_model_video_aggregation(model, loader, device):
    """Test with video-level aggregation."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for frames, label in loader:
            # frames shape: [1, C, T, H, W] -> need [T, C, H, W]
            frames = frames.squeeze(0).permute(1, 0, 2, 3)  # [T, C, H, W]
            frames = frames.to(device)
            
            # Get predictions for all frames
            outputs = model(frames)  # [T, num_classes]
            
            # Aggregate by averaging softmax probabilities
            probs = torch.softmax(outputs, dim=1)  # [T, num_classes]
            avg_probs = probs.mean(dim=0)  # [num_classes]
            
            # Final prediction
            predicted = avg_probs.argmax()
            
            total += 1
            correct += (predicted == label.to(device)).item()
    
    accuracy = 100. * correct / total
    return accuracy