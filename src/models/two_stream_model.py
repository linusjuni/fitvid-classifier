import torch
import torch.nn as nn
from torchvision import models


class SpatialStreamCNN(nn.Module):
    """
    Spatial stream for two-stream network.
    Processes single RGB frames to capture appearance information.
    
    This is essentially the same as AggregationModel2D but designed
    for the two-stream architecture.
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
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [batch, 3, H, W] (single RGB frames)
        
        Returns:
            Class scores [batch, num_classes]
        """
        return self.model(x)


class TemporalStreamCNN(nn.Module):
    """
    Temporal stream for two-stream network.
    Processes stacked optical flows to capture motion information.
    
    Uses ResNet18 architecture but modifies the first conv layer to accept
    18 input channels (9 optical flows × 2 channels for x,y flow).
    """
    
    def __init__(self, num_classes=10, pretrained=True, freeze_backbone=False):
        """
        Args:
            num_classes (int): Number of action classes
            pretrained (bool): Use ImageNet pretrained ResNet (weights for first conv will be modified)
            freeze_backbone (bool): Freeze backbone and only train classifier
        """
        super().__init__()
        
        # Start with ResNet18
        self.model = models.resnet18(pretrained=pretrained)
        
        # Modify first conv layer: 3 channels -> 18 channels
        # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # New: Conv2d(18, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.conv1 = nn.Conv2d(
            18, 64, 
            kernel_size=7, 
            stride=2, 
            padding=3, 
            bias=False
        )
        # Note: New conv1 is randomly initialized
        
        # Freeze backbone if requested (after modifying conv1)
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
            x: Input tensor [batch, 18, H, W] (stacked optical flows)
        
        Returns:
            Class scores [batch, num_classes]
        """
        return self.model(x)


class TwoStreamNetwork(nn.Module):
    """
    Two-stream network with late fusion.
    Combines predictions from spatial and temporal streams.
    """
    
    def __init__(
        self, 
        num_classes=10, 
        spatial_model=None, 
        temporal_model=None,
        fusion_weights=None
    ):
        """
        Args:
            num_classes (int): Number of action classes
            spatial_model: Pre-trained spatial stream model (optional)
            temporal_model: Pre-trained temporal stream model (optional)
            fusion_weights: (spatial_weight, temporal_weight) for weighted fusion
                           Default is (0.5, 0.5) for equal averaging
        """
        super().__init__()
        
        # Initialize streams
        if spatial_model is not None:
            self.spatial_stream = spatial_model
        else:
            self.spatial_stream = SpatialStreamCNN(num_classes=num_classes)
        
        if temporal_model is not None:
            self.temporal_stream = temporal_model
        else:
            self.temporal_stream = TemporalStreamCNN(num_classes=num_classes)
        
        # Fusion weights
        if fusion_weights is None:
            self.spatial_weight = 0.5
            self.temporal_weight = 0.5
        else:
            self.spatial_weight, self.temporal_weight = fusion_weights
    
    def forward(self, rgb_frame, flow_stack):
        """
        Args:
            rgb_frame: [batch, 3, H, W] - single RGB frames
            flow_stack: [batch, 18, H, W] - stacked optical flows
        
        Returns:
            Combined class scores [batch, num_classes]
        """
        # Get predictions from both streams
        spatial_scores = self.spatial_stream(rgb_frame)
        temporal_scores = self.temporal_stream(flow_stack)
        
        # Late fusion: weighted average of predictions
        combined_scores = (
            self.spatial_weight * spatial_scores + 
            self.temporal_weight * temporal_scores
        )
        
        return combined_scores
    
    def forward_spatial(self, rgb_frame):
        """Get spatial stream predictions only."""
        return self.spatial_stream(rgb_frame)
    
    def forward_temporal(self, flow_stack):
        """Get temporal stream predictions only."""
        return self.temporal_stream(flow_stack)
