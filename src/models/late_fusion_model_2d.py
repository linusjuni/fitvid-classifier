import torch.nn as nn
from torchvision import models


class LateFusionModel2D(nn.Module):
    """
    Late fusion 2D CNN for video classification.
    
    Extracts features from each frame independently using ResNet18 backbone,
    then pools features across time and classifies.
    """
    
    def __init__(self, num_classes=10, pretrained=True, freeze_backbone=False):
        """
        Args:
            num_classes (int): Number of action classes
            pretrained (bool): Use ImageNet pretrained weights
            freeze_backbone (bool): Freeze backbone and only train classifier
        """
        super().__init__()
        
        # Load ResNet18 and remove final FC layer
        resnet = models.resnet18(pretrained=pretrained)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        
        # Feature dimension from ResNet18
        feature_dim = 512
        
        # Classifier (always trainable)
        self.classifier = nn.Linear(feature_dim, num_classes)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [batch, C, T, H, W] where T=10 frames
        
        Returns:
            Class scores [batch, num_classes]
        """
        batch_size, C, T, H, W = x.shape
        
        # Reshape to process all frames: [batch*T, C, H, W]
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # [batch, T, C, H, W]
        x = x.view(batch_size * T, C, H, W)
        
        # Extract features for all frames: [batch*T, 512, 1, 1]
        features = self.feature_extractor(x)
        
        # Reshape back: [batch, T, 512, 1, 1]
        features = features.view(batch_size, T, -1)
        
        # Average pool over time: [batch, 512]
        pooled_features = features.mean(dim=1)
        
        # Classify: [batch, num_classes]
        output = self.classifier(pooled_features)
        
        return output