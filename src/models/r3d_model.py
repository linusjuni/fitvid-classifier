import torch.nn as nn
from torchvision import models


class R3DModel(nn.Module):
    """
    3D CNN for video classification using R3D-18.
    
    Uses 3D convolutions to process spatial and temporal dimensions simultaneously.
    Pretrained on Kinetics-400 action recognition dataset.
    """
    
    def __init__(self, num_classes=10, pretrained=True, freeze_backbone=False):
        """
        Args:
            num_classes (int): Number of action classes
            pretrained (bool): Use Kinetics-400 pretrained weights
            freeze_backbone (bool): Freeze backbone and only train classifier
        """
        super().__init__()
        
        # Load R3D-18 pretrained on Kinetics-400
        self.model = models.video.r3d_18(pretrained=pretrained)
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Replace final layer for our number of classes
        # Always keep this trainable even if backbone is frozen
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [batch, C, T, H, W] where T=10 frames
        
        Returns:
            Class scores [batch, num_classes]
        """
        return self.model(x)