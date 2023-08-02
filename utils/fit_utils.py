"""
Functions specific to the fitting pipeline
"""
from typing import Optional, List, Union, Dict

import pandas as pd
import numpy as np
import scipy.io
import h5py
import cornet
import torch
import torchvision.models as models

# available models
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge

import sys

sys.path.append("..")

from paths import *
from constants import *

from spacetorch.spatial_RN18 import SpatialResNet18

sys.path.append(UTILS_PATH)
import regression_utils as rutils

checkpoint_mapping = {
    0.0: "/oak/stanford/groups/kalanit/biac2/kgs/projects/spacenet/spacetorch/vissl_checkpoints/resnet18_simclr_checkpoints_bs128x4/model_phase145.torch",
    0.1: "/oak/stanford/groups/kalanit/biac2/kgs/projects/spacenet/spacetorch/vissl_checkpoints/simclr_spatial_resnet18_fuzzy_swappedon_SineGrating2019_lw01_checkpoints/model_phase145.torch",
    0.25: "/oak/stanford/groups/kalanit/biac2/kgs/projects/spacenet/spacetorch/vissl_checkpoints/simclr_spatial_resnet18_fuzzy_swappedon_SineGrating2019_checkpoints/model_phase145.torch",
    1.25: "/oak/stanford/groups/kalanit/biac2/kgs/projects/spacenet/spacetorch/vissl_checkpoints/simclr_spatial_resnet18_fuzzy_swappedon_SineGrating2019_lwx5_checkpoints/model_phase145.torch",
}


def get_indices(subj: str, shared: bool = False):

    order = scipy.io.loadmat(BETA_PATH + "datab3nativesurface_subj" + subj)
    data = pd.read_csv(
        NSDDATA_PATH + "ppdata/subj" + subj + "/behav/responses.tsv", sep="\t"
    )
    expdesign = scipy.io.loadmat(NSDDATA_PATH + "experiments/nsd/nsd_expdesign.mat")

    # 73KIDs
    all_ids = np.array(data["73KID"])
    vals, idx_start, count = np.unique(all_ids, return_counts=True, return_index=True)
    which_reps = vals[count == N_REPEATS]
    mask_3reps = np.isin(all_ids, which_reps)
    id_nums_3reps = np.array(data["73KID"])[mask_3reps]
    rep_vals = np.unique(id_nums_3reps)  # sorted version of beta order

    # how the betas are ordered (using COCO 73K id numbers)
    beta_order_in_73Kids = all_ids[order["allixs"][0] - 1]
    beta_mask = np.isin(
        beta_order_in_73Kids, id_nums_3reps
    )  # mask just those images with 3 repeats in the cases where subjects did not finish the full experiment
    beta_order = (
        beta_order_in_73Kids[beta_mask] - 1
    )  # -1 to convert from matlab to python indexing

    if shared:  # use shared 515 across all subjects for val mask
        val_ids = h5py.File(
            LOCALDATA_PATH + "ventral_rois/shared_73Kids.h5",
            "r",
        )
        sharedix = np.array(val_ids["shared_515"])
    else:
        # shared (i.e. validation) IDS (but include all potential shared reps for the subj, not min across subjs)
        sharedix = expdesign["sharedix"][0]
    validation_mask = np.isin(rep_vals, sharedix)

    return beta_order, beta_mask, validation_mask


def get_model(model_name: str, pretrained: bool = True, spatial_weight: float = 1.25):

    if model_name == "alexnet_torch":
        model = models.alexnet(pretrained=pretrained)
    elif model_name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=pretrained)
    elif model_name == "resnet101":
        model = models.resnet101(pretrained=pretrained)
    elif model_name == "vgg16":
        model = models.vgg16(pretrained=pretrained)
    elif model_name == "cornet-s":
        device = None if torch.cuda.is_available() else "cpu"
        model = getattr(cornet, "cornet_s")(pretrained=pretrained, map_location=device)
        model = model.module
    elif model_name == "slowfast" or model_name == "slowfast_full":
        model = torch.hub.load(
            "facebookresearch/pytorchvideo", "slowfast_r50", pretrained=pretrained
        )
    elif model_name == "spacetorch":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        weight_path = checkpoint_mapping[float(spatial_weight)]
        ckpt = torch.load(weight_path, map_location=torch.device(device))
        model_params = ckpt["classy_state_dict"]["base_model"]["model"]["trunk"]
        model = SpatialResNet18()
        model.load_state_dict(model_params, strict=False)

    return model


def get_model_layers(model_name: str, model_layer_strings: Union[str, List[str]]):

    if model_layer_strings is not None:
        if isinstance(model_layer_strings, str):
            if "," in model_layer_strings:
                model_layer_strings = model_layer_strings.split(",")
            else:
                model_layer_strings = [model_layer_strings]
    else:
        if model_name == "alexnet":
            model_layer_strings = ALEXNET_LAYERS
        elif model_name == "alexnet_torch":
            model_layer_strings = ALEXNET_TORCH_LAYERS
        elif model_name == "resnet18":
            model_layer_strings = RESNET18_LAYERS
        elif model_name == "resnet50":
            model_layer_strings = RESNET50_LAYERS
        elif model_name == "resnet101":
            model_layer_strings = RESNET101_LAYERS
        elif model_name == "vgg16":
            model_layer_strings = VGG16_LAYERS
        elif model_name == "cornet-s":
            model_layer_strings = CORNETS_LAYERS
        elif model_name == "slowfast":
            model_layer_strings = SLOWFAST_LAYERS
        elif model_name == "slowfast_full":
            model_layer_strings = MATCHING_SLOWFAST_LAYERS
        elif model_name == "spacetorch":
            model_layer_strings = SPACETORCH_LAYERS

    model_layer_strings = [
        item
        for sublist in [
            [item] if type(item) is not list else item for item in model_layer_strings
        ]
        for item in sublist
    ]

    return model_layer_strings


def mapping(features, Y, splits, layer_keys, mapping_func, CV, return_weights=False):

    map_args = None
    gridcv_params = None

    # map from features to voxel responses
    if mapping_func == "PLS":
        map_class = PLSRegression
        if CV == 0:
            map_args = {"n_components": 25, "scale": False}
        elif CV == 1:
            gridcv_params = {
                "n_components": [2, 5, 10, 25, 50, 100],
                "scale": [False],
            }
    elif mapping_func == "Ridge":
        map_class = Ridge
        if CV == 0:
            map_args = {"alpha": 100000}
        elif CV == 1:
            gridcv_params = {
                "alpha": list(np.linspace(5000, 100000, 10)),
                "fit_intercept": [True],  # all past CVs have returned True - 02/10/22
            }
    else:  # not instantiated yet
        raise ValueError(f"Mapping function: {mapping_func} not recognized")

    all_resdict = {}
    for l in layer_keys:  # for each layer ...
        print("evaluating %s" % l)
        feats = features[l]

        res = rutils.train_and_test_scikit_regressor(
            features=feats,
            labels=Y,
            splits=splits,
            model_class=map_class,
            model_args=map_args,
            gridcv_params=gridcv_params,
            feature_norm=False,
            return_models=True,
        )
        if CV == 1:
            print([_m.best_params_ for _m in res["models"]])  # list winners

        all_resdict[l] = res

    rsquared_array = {}
    for l in layer_keys:
        rsquared_array[l] = all_resdict[l]["test"]["mean_rsquared_array"]

    if return_weights:
        print("yes")
        all_weights = {}
        for l in layer_keys:
            all_weights[l] = [
                all_resdict[l]["models"][s].coef_ for s in range(len(splits))
            ]
        return rsquared_array, all_weights

    return rsquared_array
