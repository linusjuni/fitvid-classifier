import torch.nn as nn
from torchvision import models


class AggregationModel2D(nn.Module):
    """
    Per-frame 2D CNN baseline for video classification.
    
    Training: Processes individual frames independently.
    Testing: Averages predictions across all frames in a video.
    """
    
    def __init__(self, num_classes=10, pretrained=True, freeze_backbone=False):
        """
        Args:
            num_classes (int): Number of action classes
            pretrained (bool): Use ImageNet pretrained weights
            freeze_backbone (bool): Freeze backbone and only train classifier
        """
        super().__init__()
        self.model = models.resnet18(pretrained=pretrained)
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Replace final layer for our number of classes
        # Always keep this trainable even if backbone is frozen
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [batch, 3, H, W] (single frames)
        
        Returns:
            Class scores [batch, num_classes]
        """
        return self.model(x)