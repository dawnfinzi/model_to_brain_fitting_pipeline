import sys

sys.path.append("..")
from paths import *
from typing import Optional
import torch
import torchvision
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.transforms import (
    ShortSideScale,
    UniformTemporalSubsample,
)
import h5py
import numpy as np
from PIL import Image


class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """

    def __init__(self, slowfast_alpha):
        super().__init__()
        self.slowfast_alpha = slowfast_alpha

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // self.slowfast_alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list


norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
video_norm_cfg = dict(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])

# SlowFast network training details
FRAME_LENGTH = 8
SAMPLE_RATE = 8

NSD_TRANSFORMS = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(**norm_cfg),
    ]
)

NSD_VIDEO_TRANSFORMS = torchvision.transforms.Compose(
    [
        UniformTemporalSubsample(32),
        torchvision.transforms.Lambda(lambda x: x / 255.0),
        NormalizeVideo(**video_norm_cfg),
        ShortSideScale(size=256),
        CenterCropVideo(256),
        PackPathway(4),
    ]
)


class NSDataset(torch.utils.data.Dataset):
    """NSD stimuli."""

    def __init__(self, stim_path, video=False, transform=None):
        """
        Args:
            stim_path (string): Path to the hdf file with stimuli.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        stim = h5py.File(stim_path, "r")  # 73k images
        self.data = stim["imgBrick"]
        self.video = video
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]

        if self.transform:
            if self.video:
                x = np.expand_dims(x, axis=-1)
                x = np.tile(x, FRAME_LENGTH * SAMPLE_RATE)
                x = np.transpose(x, (2, 3, 0, 1))
                x = torch.from_numpy(x)
            else:
                x = Image.fromarray(x.astype(np.uint8))
            x = self.transform(x)

        return x

    def __len__(self):
        return len(self.data)


def nsd_dataloader(
    indices: Optional[list] = None,
    video: bool = False,
    batch_size: int = 32,
) -> torch.utils.data.DataLoader:

    full_stim_path = STIM_PATH + "nsd_stimuli.hdf5"
    if video:
        transform = NSD_VIDEO_TRANSFORMS
    else:
        transform = NSD_TRANSFORMS
    dataset = NSDataset(full_stim_path, video, transform)
    if indices:
        dataset = torch.utils.data.Subset(dataset, indices)
    data_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True
    )
    return data_loader
