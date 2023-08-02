"""
This computes fits for the sorted ventral dataset (sent to Chengxu and Violet)
"""

from typing import Optional, List, Union
import argparse
import numpy as np
import h5py

import sys

sys.path.append("..")

from paths import *
from constants import *

sys.path.append(CODE_PATH)
import utils.regression_utils as rutils
from fit import extract_and_fit


def main(
    subj,
    hemi,
    area,
    model_name,
    model_layer_strings,
    subsample,
    mapping_func,
    CV,
    num_splits,
):

    all_betas = h5py.File(
        LOCALDATA_PATH
        + "ventral_rois/subj"
        + subj
        + "/"
        + hemi
        + "_zscored_betas_org_by_ROI.h5",
        "r",
    )
    avg_roi_betas = np.mean(
        np.array(all_betas[area]), axis=2
    )  # average across all 3 trials
    avg_roi_betas = np.flip(avg_roi_betas, axis=0)
    stim_ids = np.array(all_betas["stimulus_73KID"])
    beta_order = stim_ids - 1

    val_ids = h5py.File(
        LOCALDATA_PATH + "ventral_rois/shared_73Kids.h5",
        "r",
    )

    # indexing
    validation_mask = np.isin(stim_ids, np.array(val_ids["shared_515"]))
    num_train = int((8 / 9) * (avg_roi_betas.shape[0] - np.sum(validation_mask)))
    num_test = int((1 / 9) * (avg_roi_betas.shape[0] - np.sum(validation_mask)))

    # all splits for regression training
    all_splits = rutils.get_splits(
        data=avg_roi_betas,
        split_index=0,
        num_splits=num_splits,
        num_per_class_test=num_test,
        num_per_class_train=num_train,
        exclude=validation_mask,
    )

    rsquared_array = extract_and_fit(
        subj,
        model_name,
        model_layer_strings,
        subsample,
        mapping_func,
        CV,
        avg_roi_betas,
        beta_order,
        all_splits,
    )

    # save to local data folder
    h5f = h5py.File(
        RESULTS_PATH
        + "ventral_fits_by_area/subj"
        + subj
        + "_"
        + hemi
        + "_"
        + area
        + "_"
        + model_name
        + "_"
        + mapping_func
        + "_subsample_"
        + str(subsample)
        + "_"
        + str(CV)
        + "CV_fits.hdf5",
        "w",
    )

    for k, v in rsquared_array.items():
        print(str(k))
        h5f.create_dataset(str(k), data=v)
    h5f.close()


if __name__ == "__main__":
    # Parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--subj", type=str)
    parser.add_argument("--hemi", type=str, default="rh")
    parser.add_argument("--area", type=str, default="V1v")
    parser.add_argument("--model_name", type=str, default="alexnet")
    parser.add_argument(
        "--model_layer_strings", type=Optional[Union[str, List[str]]], default=None
    )
    parser.add_argument(
        "--subsample", type=int, default=1
    )  # subsample the features randomly (1) or using PCA (2)
    parser.add_argument(
        "--mapping_func", type=str, default="PLS"
    )  # which mapping function to use
    parser.add_argument("--CV", type=int, default=0)  # cross-validated (1) or not (0)
    parser.add_argument(
        "--num_splits", type=int, default=1
    )  # number of splits to average over
    ARGS, _ = parser.parse_known_args()

    main(
        ARGS.subj,
        ARGS.hemi,
        ARGS.area,
        ARGS.model_name,
        ARGS.model_layer_strings,
        ARGS.subsample,
        ARGS.mapping_func,
        ARGS.CV,
        ARGS.num_splits,
    )
