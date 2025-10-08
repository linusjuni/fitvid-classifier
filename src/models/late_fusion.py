"""ResNet late fusion model that aggregates features across frames for video-level classification."""
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

class ResNetLateFusion(nn.Module):
    def __init__(self, num_classes=10, pretrained=True, freeze_backbone=False):
        super().__init__()

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base = resnet18(weights=weights)

        self.backbone = nn.Sequential(*list(base.children())[:-2])
        self.feat_dim = base.fc.in_features

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.head = nn.Linear(self.feat_dim, num_classes)
    
    def forward(self, x):
      B, T, C, H, W = x.shape
      x = x.view(B*T, C, H, W)              # run CNN per-frame
      f = self.backbone(x)                  # (B*T, D, H', W')
      f = F.adaptive_avg_pool2d(f, 1)       # spatial avg → (B*T, D, 1, 1)
      f = f.view(B, T, self.feat_dim)       # regroup frames
      f = f.mean(dim=1)                     # temporal avg → (B, D)
      return self.head(f)                   # (B, num_classes)