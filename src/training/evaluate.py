"""Evaluation utilities."""
import torch

def test_model(model, loader, device):
    """Final test set evaluation."""
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