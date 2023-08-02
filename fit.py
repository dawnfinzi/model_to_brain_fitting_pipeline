from datetime import datetime
import numpy as np
import h5py
from sklearn.decomposition import PCA

import sys

sys.path.append("..")

from paths import *
from constants import *

sys.path.append(CODE_PATH)
from utils.memory_utils import display_top, memory_monitor
from queue import Queue
from threading import Thread

from utils.fit_utils import get_model, get_model_layers, mapping
from datasets.nsd import nsd_dataloader
from datasets.imagenet import imagenet_validation_dataloader
from feature_extractor import get_features_from_layer


def log(to_print: str):
    print(f"\nLOG: {to_print}")


def extract_and_fit(
    subj,
    model_name,
    model_layer_strings,
    subsample,
    mapping_func,
    CV,
    sorted_betas,
    beta_order,
    all_splits,
    pretrained=1,
    reduce_temporal_dims=0,
    return_weights=False,
    spatial_weight=1.25,
):

    rng = np.random.RandomState(seed=0)

    log(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log("Setting/getting model")

    # load model (and/or get features)
    pt = True if pretrained == 1 else False

    model_name = model_name.lower()
    model = get_model(model_name, pretrained=pt, spatial_weight=spatial_weight)
    model_layer_strings = get_model_layers(model_name, model_layer_strings)

    presaved = False
    if model_name == "alexnet":
        presaved = True
        num_batches = 5

    two_pathway = False
    video = False
    reduction_list = None
    if model_name == "slowfast" or model_name == "slowfast_full":
        two_pathway = True
        video = True
        if reduce_temporal_dims:
            reduction_list = np.tile(2, len(model_layer_strings))
            for lidx, l in enumerate(model_layer_strings):
                if l == "blocks.5" or l == "blocks.6.proj":  # no temporal dim to reduce
                    reduction_list[lidx] = -1

    n_val_images = 1000
    if subsample == 2:  # PCA
        imagenet_batch = imagenet_validation_dataloader(
            list(range(0, n_val_images)), video=video
        )
        imagenet_feats = get_features_from_layer(
            model,
            imagenet_batch,
            model_layer_strings,
            two_pathway=two_pathway,
            reduction_list=reduction_list,
            vectorize=True,
        )
        del imagenet_batch

    features = {}
    flat_features = {}
    final_features = {}
    n_feats_per_layer = {}
    N = n_val_images  # num features to keep if subsampling (used to be 5k for subsamp = 1, and 1k for subsamp = 2)

    if presaved:  # load precomputed features (already organized by subject)
        log("Loading saved features")
        for b in range(num_batches):
            print(b)
            h5f = h5py.File(
                FEATS_PATH
                + model
                + "/"
                + model
                + "_features_subj"
                + subj
                + "_batch"
                + str(b + 1)
                + ".h5",
                "r",
            )
            for l in model_layer_strings:
                if b == 0:
                    features[l] = h5f[l][:]
                else:
                    features[l] = np.concatenate(
                        (features[l], h5f[l][:])
                    )  # features for the training images
            h5f.close()

        for l in model_layer_strings:
            shp = features[l].shape
            flat_features[l] = features[l].reshape((shp[0], np.prod(shp[1:])))
        del features

        if subsample == 1:
            for l in model_layer_strings:
                n_feats_per_layer[l] = flat_features[l].shape[
                    1
                ]  # num feats pre subsamp
                perm = rng.permutation(
                    n_feats_per_layer[l]
                )  # pick a permutation of the set [0, ... n-1]
                keep_inds = perm[:N]  # keep the first N random features
                final_features[l] = flat_features[l][:, keep_inds]
        else:
            final_features = flat_features
        del flat_features

    else:  # compute features on the fly

        log("Computing features")
        nsd_batches = 146  # num batches to use (low memory load)
        imgs_per_batch = 500
        subj_stim_idx = np.sort(beta_order)
        keep_inds = {}
        pca_model = {}
        prev_batch_end = 0

        for b in range(nsd_batches):
            log(b)
            subj_batch_idx = subj_stim_idx[
                (subj_stim_idx >= imgs_per_batch * (b))
                & (subj_stim_idx < imgs_per_batch * (b + 1))
            ]
            batch_end = len(subj_batch_idx)

            batch = nsd_dataloader(
                list(subj_batch_idx),
                video=video,
                batch_size=len(list(subj_batch_idx)),
            )

            batch_feats = get_features_from_layer(
                model,
                batch,
                model_layer_strings,
                two_pathway=two_pathway,
                reduction_list=reduction_list,
                batch_size=len(list(subj_batch_idx)),
                vectorize=True,
            )

            for l in model_layer_strings:
                if b == 0:
                    n_feats_per_layer[l] = batch_feats[l].shape[
                        1
                    ]  # num feats pre subsamp
                    print(n_feats_per_layer[l])
                    perm = rng.permutation(
                        n_feats_per_layer[l]
                    )  # pick a permutation of the set [0, ... n-1]
                    keep_inds[l] = perm[:N]
                    feat_length = (
                        n_feats_per_layer[l]
                        if subsample != 1
                        else min(N, n_feats_per_layer[l])
                    )

                    if (
                        subsample == 2
                    ):  # on first batch, PCA imagenet feats and keep models
                        feat_length = np.min([n_feats_per_layer[l], n_val_images])
                        pca_model[l] = PCA(n_components=feat_length)
                        pca_model[l].fit(imagenet_feats[l])
                        del imagenet_feats[l]

                    # preallocate array
                    final_features[l] = np.zeros(
                        (subj_stim_idx.shape[0], feat_length), dtype=np.float32
                    )

                if subsample == 0:  # no subsampling
                    final_features[l][
                        prev_batch_end : prev_batch_end + batch_end, :
                    ] = batch_feats[l]
                elif subsample == 1:  # random subsampling
                    final_features[l][
                        prev_batch_end : prev_batch_end + batch_end, :
                    ] = batch_feats[l][:, keep_inds[l]]
                elif subsample == 2:  # PCA
                    final_features[l][
                        prev_batch_end : prev_batch_end + batch_end, :
                    ] = pca_model[l].transform(batch_feats[l])

            prev_batch_end += batch_end
            del batch, batch_feats

    # map from features to voxel responses
    log("Mapping to voxel responses")
    rsquared_array = mapping(
        final_features,
        sorted_betas,
        all_splits,
        model_layer_strings,
        mapping_func,
        CV,
        return_weights,
    )

    return rsquared_array
