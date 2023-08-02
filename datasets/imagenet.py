from typing import Optional
from pathlib import Path

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


NUM_TRAIN_IMAGES = 1_281_167
NUM_VALIDATION_IMAGES = 50_000

# SlowFast network training details
FRAME_LENGTH = 8
SAMPLE_RATE = 8

norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
video_norm_cfg = dict(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])

IMAGENET_TRANSFORMS = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(**norm_cfg),
    ]
)

IMAGENET_VIDEO_TRANSFORMS = torchvision.transforms.Compose(
    [
        UniformTemporalSubsample(32),
        torchvision.transforms.Lambda(lambda x: x / 255.0),
        NormalizeVideo(**video_norm_cfg),
        ShortSideScale(size=256),
        CenterCropVideo(256),
        PackPathway(4),
    ]
)


class ImageNetData(torchvision.datasets.ImageFolder):
    """ImageNet data"""

    def __init__(self, root, video=False, transform=None):
        super(ImageNetData, self).__init__(root, transform)
        self.video = video

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            image
        """
        path, _ = self.imgs[index]
        img = self.loader(path)
        if self.video:
            img = np.asarray(img)
            img = np.expand_dims(img, axis=-1)
            img = np.tile(img, FRAME_LENGTH * SAMPLE_RATE)
            img = np.transpose(img, (2, 3, 0, 1))
            img = torch.from_numpy(img)

        if self.transform is not None:
            img = self.transform(img)

        return img


def imagenet_validation_dataloader(
    indices: Optional[list] = None,
    video: bool = False,
    batch_size: int = 32,
) -> torch.utils.data.DataLoader:

    imagenet_val_dir = (
        "/oak/stanford/groups/kalanit/biac2/kgs/ml_datasets/imagenet/validation/"
    )
    assert Path(imagenet_val_dir).exists()
    if video:
        transform = IMAGENET_VIDEO_TRANSFORMS
    else:
        transform = IMAGENET_TRANSFORMS
    dataset = ImageNetData(imagenet_val_dir, video, transform)
    if indices:
        dataset = torch.utils.data.Subset(dataset, indices)
    data_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True
    )
    return data_loader
