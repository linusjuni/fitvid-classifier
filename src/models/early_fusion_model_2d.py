import torch
import torch.nn as nn
from torchvision import models


class EarlyFusionModel2D(nn.Module):
    """
    Early fusion 2D CNN for video classification.
    
    Stacks all T frames as input channels (3*T channels total),
    processes with ResNet18 where first conv layer is modified.
    """
    
    def __init__(self, num_classes=10, num_frames=10, pretrained=True, freeze_backbone=False):
        """
        Args:
            num_classes (int): Number of action classes
            num_frames (int): Number of frames to stack (T)
            pretrained (bool): Use ImageNet pretrained weights
            freeze_backbone (bool): Freeze backbone and only train first conv + classifier
        """
        super().__init__()
        self.num_frames = num_frames
        
        # Load pretrained ResNet18
        resnet = models.resnet18(pretrained=pretrained)
        
        # Modify first conv layer to accept 3*T channels
        original_conv1 = resnet.conv1
        in_channels = 3 * num_frames  # 3 RGB channels * T frames
        
        # Create new conv layer with more input channels
        self.conv1 = nn.Conv2d(
            in_channels,
            original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=False
        )
        
        # Initialize with pretrained weights (repeated across frames with noise)
        if pretrained:
            with torch.no_grad():
                # Repeat pretrained weights across all frames
                pretrained_weight = original_conv1.weight  # [64, 3, 7, 7]
                repeated_weight = pretrained_weight.repeat(1, num_frames, 1, 1) / num_frames
                
                # Add small random noise to break symmetry
                repeated_weight += torch.randn_like(repeated_weight) * 0.01
                
                self.conv1.weight.copy_(repeated_weight)
        
        # Keep rest of ResNet18 architecture
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        
        # Freeze backbone if requested (except first conv and final fc)
        if freeze_backbone:
            for param in [self.bn1, self.layer1, self.layer2, 
                         self.layer3, self.layer4]:
                for p in param.parameters():
                    p.requires_grad = False
        
        # Replace final layer for our number of classes (always trainable)
        self.fc = nn.Linear(resnet.fc.in_features, num_classes)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [batch, C, T, H, W] where T=num_frames
        
        Returns:
            Class scores [batch, num_classes]
        """
        batch_size, C, T, H, W = x.shape
        
        # Reshape to stack frames as channels: [batch, C*T, H, W]
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # [batch, T, C, H, W]
        x = x.view(batch_size, C * T, H, W)
        
        # Forward through modified ResNet18
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x