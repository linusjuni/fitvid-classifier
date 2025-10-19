import os
from glob import glob
import pandas as pd
from PIL import Image
import torch
import numpy as np
from torchvision import transforms as T

# Dataset paths
DATASET_PATHS = {
    "leakage": "/dtu/datasets1/02516/ufc10",
    "no_leakage": "/dtu/datasets1/02516/ucf101_noleakage",
}


def get_data_root(dataset_name="no_leakage"):
    """Get the root directory for a dataset."""
    if dataset_name not in DATASET_PATHS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Choose from {list(DATASET_PATHS.keys())}"
        )
    return DATASET_PATHS[dataset_name]


class FlowTransform:
    """Custom transform for multi-channel optical flow."""
    
    def __init__(self, size=(224, 224), normalize=True):
        self.size = size
        self.normalize = normalize
        # Pre-computed statistics from training data
        self.mean = 0.9352
        self.std = 0.1626
    
    def __call__(self, flow_array):
        """
        Args:
            flow_array: numpy array of shape (H, W, 18)
        
        Returns:
            torch tensor of shape (18, H', W') normalized
        """
        # Resize each channel individually
        h, w, c = flow_array.shape
        
        if (h, w) != self.size:
            # Use PIL to resize each channel
            resized_channels = []
            for i in range(c):
                channel = Image.fromarray(flow_array[:, :, i])
                channel_resized = channel.resize(self.size, Image.BILINEAR)
                resized_channels.append(np.array(channel_resized))
            flow_array = np.stack(resized_channels, axis=-1)
        
        # Convert to tensor: (H, W, 18) -> (18, H, W)
        flow_tensor = torch.from_numpy(flow_array).float()
        flow_tensor = flow_tensor.permute(2, 0, 1)
        
        # Normalize to [0, 1]
        flow_tensor = flow_tensor / 255.0
        
        # Standardize using pre-computed statistics
        if self.normalize:
            flow_tensor = (flow_tensor - self.mean) / self.std
        
        return flow_tensor


class FrameImageDataset(torch.utils.data.Dataset):
    """Dataset that returns individual frames from videos."""

    def __init__(self, dataset_name="no_leakage", split="train", transform=None):
        root_dir = get_data_root(dataset_name)
        self.frame_paths = sorted(glob(f"{root_dir}/frames/{split}/*/*/*.jpg"))
        self.df = pd.read_csv(f"{root_dir}/metadata/{split}.csv")
        self.split = split
        self.transform = transform

    def __len__(self):
        return len(self.frame_paths)

    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx):
        frame_path = self.frame_paths[idx]
        video_name = frame_path.split("/")[-2]
        video_meta = self._get_meta("video_name", video_name)
        label = video_meta["label"].item()

        frame = Image.open(frame_path).convert("RGB")

        if self.transform:
            frame = self.transform(frame)
        else:
            frame = T.ToTensor()(frame)

        return frame, label


class FrameVideoDataset(torch.utils.data.Dataset):
    """Dataset that returns all frames from a video."""

    def __init__(
        self,
        dataset_name="no_leakage",
        split="train",
        transform=None,
        stack_frames=True,
    ):
        root_dir = get_data_root(dataset_name)
        self.video_paths = sorted(glob(f"{root_dir}/videos/{split}/*/*.avi"))
        self.df = pd.read_csv(f"{root_dir}/metadata/{split}.csv")
        self.split = split
        self.transform = transform
        self.stack_frames = stack_frames
        self.n_sampled_frames = 10

    def __len__(self):
        return len(self.video_paths)

    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video_name = video_path.split("/")[-1].split(".avi")[0]
        video_meta = self._get_meta("video_name", video_name)
        label = video_meta["label"].item()

        video_frames_dir = video_path.replace(".avi", "").replace("videos", "frames")
        video_frames = self.load_frames(video_frames_dir)

        if self.transform:
            frames = [self.transform(frame) for frame in video_frames]
        else:
            frames = [T.ToTensor()(frame) for frame in video_frames]

        if self.stack_frames:
            frames = torch.stack(frames).permute(1, 0, 2, 3)  # [C, T, H, W]

        return frames, label

    def load_frames(self, frames_dir):
        frames = []
        for i in range(1, self.n_sampled_frames + 1):
            frame_file = os.path.join(frames_dir, f"frame_{i}.jpg")
            frame = Image.open(frame_file).convert("RGB")
            frames.append(frame)
        return frames


class DualStreamDataset(torch.utils.data.Dataset):
    """
    Dataset for two-stream networks that returns:
    - Single RGB frame (for spatial stream)
    - Stacked optical flows (for temporal stream)
    - Label
    
    Handles the mismatch: 10 RGB frames vs 9 optical flows.
    """
    
    def __init__(
        self,
        dataset_name="no_leakage",
        split="train",
        spatial_transform=None,
        temporal_transform=None,
        frame_sample_strategy="random",  # "random", "middle", or specific index
    ):
        """
        Args:
            dataset_name: "leakage" or "no_leakage"
            split: "train", "val", or "test"
            spatial_transform: transforms for RGB frame (spatial stream)
            temporal_transform: transforms for optical flow stack (temporal stream)
            frame_sample_strategy: which RGB frame to use
                - "random": random frame during training (data augmentation)
                - "middle": always use middle frame (frame 5)
                - int: specific frame index (1-10)
        """
        root_dir = get_data_root(dataset_name)
        self.video_paths = sorted(glob(f"{root_dir}/videos/{split}/*/*.avi"))
        self.df = pd.read_csv(f"{root_dir}/metadata/{split}.csv")
        self.split = split
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.frame_sample_strategy = frame_sample_strategy
        self.n_frames = 10
        self.n_flows = 9
        
    def __len__(self):
        return len(self.video_paths)
    
    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]
    
    def _get_frame_index(self):
        """Determine which RGB frame to use based on strategy."""
        if self.frame_sample_strategy == "random":
            # Random frame (1-10) for data augmentation during training
            return torch.randint(1, self.n_frames + 1, (1,)).item()
        elif self.frame_sample_strategy == "middle":
            # Middle frame (frame 5)
            return 5
        elif isinstance(self.frame_sample_strategy, int):
            # Specific frame index
            return self.frame_sample_strategy
        else:
            raise ValueError(f"Unknown frame_sample_strategy: {self.frame_sample_strategy}")
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video_name = video_path.split("/")[-1].split(".avi")[0]
        video_meta = self._get_meta("video_name", video_name)
        label = video_meta["label"].item()
        
        # Get directories
        video_frames_dir = video_path.replace(".avi", "").replace("videos", "frames")
        video_flows_dir = video_path.replace(".avi", "").replace("videos", "flows_png")
        
        # Load single RGB frame for spatial stream (PIL Image)
        frame_idx = self._get_frame_index()
        rgb_frame = self.load_rgb_frame(video_frames_dir, frame_idx)
        
        # Load and stack all optical flows for temporal stream (numpy array)
        flow_stack = self.load_flow_stack(video_flows_dir)  # (H, W, 18)
        
        # Apply transforms
        if self.spatial_transform:
            rgb_frame = self.spatial_transform(rgb_frame)
        else:
            rgb_frame = T.ToTensor()(rgb_frame)
        
        if self.temporal_transform:
            # temporal_transform should handle numpy array -> tensor conversion
            flow_stack = self.temporal_transform(flow_stack)
        else:
            # Default: convert to tensor and normalize to [0, 1]
            flow_stack = torch.from_numpy(flow_stack).float() / 255.0
            flow_stack = flow_stack.permute(2, 0, 1)  # (H, W, 18) -> (18, H, W)
        
        return rgb_frame, flow_stack, label
    
    def load_rgb_frame(self, frames_dir, frame_idx):
        """Load a single RGB frame."""
        frame_file = os.path.join(frames_dir, f"frame_{frame_idx}.jpg")
        frame = Image.open(frame_file).convert("RGB")
        return frame
    
    def load_flow_stack(self, flows_dir):
        """
        Load all 9 optical flows and stack them.
        
        Each flow PNG has 3 channels (R, G, B) where:
        - R channel: horizontal flow (flow_x)
        - G channel: vertical flow (flow_y)  
        - B channel: unused (constant ~254)
        
        Returns:
            numpy array with shape (H, W, 18) where channels are:
            [flow1_x, flow1_y, flow2_x, flow2_y, ..., flow9_x, flow9_y]
        """
        flow_channels = []
        
        for i in range(1, self.n_flows + 1):
            flow_file = os.path.join(flows_dir, f"flow_{i}_{i+1}.png")
            flow_img = Image.open(flow_file)
            flow_array = np.array(flow_img)  # Shape: (H, W, 3)
            
            # Extract x and y flow from R and G channels
            flow_x = flow_array[:, :, 0]  # R channel
            flow_y = flow_array[:, :, 1]  # G channel
            
            flow_channels.append(flow_x)
            flow_channels.append(flow_y)
        
        # Stack all channels: (H, W, 18)
        stacked_flow = np.stack(flow_channels, axis=-1)
        
        return stacked_flow.astype(np.uint8)