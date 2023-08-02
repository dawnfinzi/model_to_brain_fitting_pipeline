"""
Constants that might be used by multiple scripts
"""

# how many times each image was shown
N_REPEATS = 3
SUBJECTS = ["01", "02", "03", "04", "05", "06", "07", "08"]

# layer names for different models
ALEXNET_LAYERS = ["conv1", "conv2", "conv3", "conv4", "conv5", "fc6", "fc7"]
ALEXNET_TORCH_LAYERS = (
    ["features.2"]
    + ["features.5"]
    + ["features.7"]
    + ["features.9"]
    + ["features.12"]
    + ["classifier.2"]
    + ["classifier.5"],
)
RESNET18_LAYERS = (
    ["relu", "maxpool"]
    + ["layer1.0", "layer1.1"]
    + ["layer2.0", "layer2.1"]
    + ["layer3.0", "layer3.1"]
    + ["layer4.0", "layer4.1"]
    + ["avgpool"]
)
RESNET50_LAYERS = (
    ["relu", "maxpool"]
    + [f"layer1.{i}" for i in range(3)]
    + [f"layer2.{i}" for i in range(4)]
    + [f"layer3.{i}" for i in range(6)]
    + [f"layer4.{i}" for i in range(3)]
    + ["avgpool"],
)
RESNET101_LAYERS = (
    ["relu", "maxpool"]
    + [f"layer1.{i}" for i in range(3)]
    + [f"layer2.{i}" for i in range(4)]
    + [f"layer3.{i}" for i in range(23)]
    + [f"layer4.{i}" for i in range(3)]
    + ["avgpool"],
)
VGG16_LAYERS = (
    ["features.4"]
    + ["features.9"]
    + ["features.16"]
    + ["features.23"]
    + ["features.30"]
    + ["classifier.1"]
    + ["classifier.4"],
)
CORNETS_LAYERS = ["V1", "V2", "V4", "IT", "decoder.avgpool"]
SLOWFAST_LAYERS = [
    "blocks.1.multipathway_blocks.0.res_blocks.2",  # slow
    "blocks.1.multipathway_blocks.1.res_blocks.2",  # fast
    "blocks.2.multipathway_blocks.0.res_blocks.2",  # slow
    "blocks.2.multipathway_blocks.1.res_blocks.2",  # fast
    "blocks.3.multipathway_blocks.0.res_blocks.2",  # slow
    "blocks.3.multipathway_blocks.1.res_blocks.2",  # fast
    "blocks.4.multipathway_blocks.0.res_blocks.2",  # slow
    "blocks.4.multipathway_blocks.1.res_blocks.2",  # fast
    "blocks.5",
    "blocks.6.proj",
]
MATCHING_SLOWFAST_LAYERS = (
    ["blocks.0.multipathway_blocks.0", "blocks.0.multipathway_blocks.1"]
    + [f"blocks.1.multipathway_blocks.0.res_blocks.{i}" for i in range(3)]  # slow
    + [f"blocks.1.multipathway_blocks.1.res_blocks.{i}" for i in range(3)]  # fast
    + [f"blocks.2.multipathway_blocks.0.res_blocks.{i}" for i in range(4)]  # slow
    + [f"blocks.2.multipathway_blocks.1.res_blocks.{i}" for i in range(4)]  # fast
    + [f"blocks.3.multipathway_blocks.0.res_blocks.{i}" for i in range(6)]  # slow
    + [f"blocks.3.multipathway_blocks.1.res_blocks.{i}" for i in range(6)]  # fast
    + [f"blocks.4.multipathway_blocks.0.res_blocks.{i}" for i in range(3)]  # slow
    + [f"blocks.4.multipathway_blocks.1.res_blocks.{i}" for i in range(3)]  # fast
    + ["blocks.5", "blocks.6.proj"]
)
SPACETORCH_LAYERS = (
    ["base_model.conv1", "base_model.maxpool"]
    + ["base_model.layer1.0", "base_model.layer1.1"]
    + ["base_model.layer2.0", "base_model.layer2.1"]
    + ["base_model.layer3.0", "base_model.layer3.1"]
    + ["base_model.layer4.0", "base_model.layer4.1"]
    + ["base_model.avgpool"]
)

# list of areas included in streams ROIs
ROI_NAMES = [
    "Unknown",
    "Early",
    "Midventral",
    "Midlateral",
    "Midparietal",
    "Ventral",
    "Lateral",
    "Parietal",
]

# Color palette for stream ROIs
ROI_COLORS = {
    "Early": "#a6a6a6",
    "Midventral": "#f4bdd8",
    "Midlateral": "#ccdaff",
    "Midparietal": "#b3ffc6",
    "Ventral": "#DC267F",
    "Lateral": "#4d7fff",
    "Parietal": "#006600",
}
