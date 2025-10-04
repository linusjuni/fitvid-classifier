"""Simple ResNet baseline for per-frame classification."""
import torch.nn as nn
from torchvision import models

class ResNetBaseline(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()
        self.model = models.resnet18(pretrained=pretrained)
        # Replace final layer for our number of classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)