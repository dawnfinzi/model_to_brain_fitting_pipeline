import argparse
from typing import List, Union
from pathlib import Path
from datetime import datetime
import torchvision.models as models

import sys

sys.path.append("..")
from paths import *
from constants import *

from utils.fit_utils import get_model_layers
from datasets.imagenet import imagenet_validation_dataloader
from datasets.nsd import nsd_dataloader
from feature_saver import FeatureSaver


def log(to_print: str):
    print(f"\nLOG: {to_print}")


def main(model_name, dataset_name, batch_num, model_layer_strings):

    # Create dataloader for image data
    indices = None
    if batch_num is not None:
        indices = list(range(1000 * (batch_num), 1000 * (batch_num + 1)))

    dataset_name = dataset_name.lower()
    if dataset_name == "nsd":
        batch = nsd_dataloader(indices)
    elif dataset_name == "imagenet":
        batch = imagenet_validation_dataloader(indices)
    else:  # not instantiated yet
        raise ValueError(f"Dataset type: {dataset_name} not recognized")

    # Load model
    model, model_layer_strings = get_model_layers(model_name, model_layer_strings)

    # use FeatureSaver to extract and save features
    save_path: Path = (
        Path(FEATS_PATH) / model_name / f"{model_name}_{dataset_name}_features_all.h5"
    )
    if batch_num is not None:
        save_path: Path = (
            Path(FEATS_PATH)
            / model_name
            / f"{model_name}_{dataset_name}_features_batch{str(batch_num)}.h5"
        )

    log(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log("Constructing feature saver")
    feature_saver: FeatureSaver = FeatureSaver(
        model, model_layer_strings, batch, save_path
    )

    log("Extracting features")
    feature_saver.compute_features()

    log("Saving features")
    feature_saver.save_features()

    log("All done!")
    log(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    # Parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--batch_num", type=int, default=None)
    parser.add_argument(
        "--model_layer_strings", type=Union[str, List[str]], default=None
    )

    ARGS, _ = parser.parse_known_args()

    main(
        ARGS.model_name,
        ARGS.dataset_name,
        ARGS.batch_num,
        ARGS.model_layer_strings,
    )
