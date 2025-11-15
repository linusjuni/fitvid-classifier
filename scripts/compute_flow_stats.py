import torch
import numpy as np
from src.datasets import DualStreamDataset, FlowTransform

# Create dataset without normalization
temporal_transform = FlowTransform(size=(224, 224))

dataset = DualStreamDataset(
    dataset_name="no_leakage",
    split="train",
    spatial_transform=None,
    temporal_transform=temporal_transform,
    frame_sample_strategy="middle",
)

print("Computing flow statistics from 100 samples...")
flow_values = []

for i in range(min(100, len(dataset))):
    _, flow, _ = dataset[i]
    flow_values.append(flow)
    if i % 20 == 0:
        print(f"  Processed {i}/100 samples...")

flow_tensor = torch.stack(flow_values)  # (100, 18, 224, 224)

print("\nFlow Statistics (per-channel):")
print(f"Mean: {flow_tensor.mean(dim=(0, 2, 3))}")
print(f"Std:  {flow_tensor.std(dim=(0, 2, 3))}")
print(f"\nOverall:")
print(f"Mean: {flow_tensor.mean():.4f}")
print(f"Std:  {flow_tensor.std():.4f}")
print(f"Min:  {flow_tensor.min():.4f}")
print(f"Max:  {flow_tensor.max():.4f}")
