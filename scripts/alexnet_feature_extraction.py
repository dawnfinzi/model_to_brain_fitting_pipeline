"""
Extracts features from AlexNet (using tensorflow v1) for a given subject's images

"""

import argparse
import pandas as pd
import numpy as np
import h5py

import tensorflow as tf

import sys

utils_dir = "/sni-storage/kalanit/Projects/Dawn/NSD/code/fit_pipeline/utils/"
sys.path.append(utils_dir)

from general_utils import norm_image
from models_alexnet import alexnet
from models_alexnet import alexnet_wrapper

# setup paths
STIM_PATH = "../../../data/nsddata_stimuli/stimuli/nsd/"
CKPT_PATH = "../models/checkpoints/alexnet/model.ckpt-115000"


def main(subjid, batch_num, total_batches):

    assert batch_num != 0, "batch indexing starts at 1!"

    subj_list = ["01", "02", "03", "04", "05", "06", "07", "08"]
    sidx = subj_list.index(subjid)

    n_repeats = 3

    # load and subset massive stimuli file
    stim = h5py.File(STIM_PATH + "nsd_stimuli.hdf5", "r")  # 73k images

    data = pd.read_csv(
        "../../../data/nsddata/ppdata/subj" + subjid + "/behav/responses.tsv", sep="\t"
    )

    all_ids = np.array(data["73KID"])
    vals, idx_start, count = np.unique(all_ids, return_counts=True, return_index=True)
    which_reps = vals[count == n_repeats]

    mask_3reps = np.isin(all_ids, which_reps)
    id_nums_3reps = np.array(data["73KID"])[mask_3reps] - 1
    adj_idx = np.unique(id_nums_3reps)

    subj_stim = stim["imgBrick"][adj_idx, :, :, :]
    del stim

    ims_per_batch = np.floor(len(np.unique(id_nums_3reps)) / total_batches)
    batch_idx = batch_num - 1

    # okay let's get started
    tf.reset_default_graph()  # just in case

    layer_keys = ["conv1", "conv2", "conv3", "conv4", "conv5", "fc6", "fc7"]
    total_outputs = {}
    
    if batch_num == total_batches: #last batch
        total_batch_images = (
        subj_stim[int(ims_per_batch * batch_idx): , :, :, :]
        / 255.0
        ) # go to end to catch remainder if total images not divisible by total batches
    else:
        total_batch_images = (
            subj_stim[int(ims_per_batch * batch_idx) : int(ims_per_batch * (batch_idx + 1)), :, :, :]
            / 255.0
        )
    
    del subj_stim

    # divide into minibatch chunks of 500
    num_minibatch = int(np.floor(ims_per_batch / 500))
    if (ims_per_batch % 500.) != 0:
        num_minibatch += 1
    print(num_minibatch)

    for idx in range(0, num_minibatch):

        print(idx)
        
        if idx == (num_minibatch - 1):
            minibatch_images = total_batch_images[500 * idx:, :, :, :]
        else:
            minibatch_images = total_batch_images[500 * idx : 500 * (idx + 1), :, :, :]

        # create image tensor
        batch_image_tensor = tf.convert_to_tensor(minibatch_images, dtype=tf.float32)
        # resize for alexnet
        batch_resized_images = tf.image.resize_images(batch_image_tensor, (224, 224))

        # initialize model
        batch_convnet = alexnet(batch_resized_images)

        with tf.Session() as sess:
            tf_saver = tf.train.Saver()
            tf_saver.restore(sess, CKPT_PATH)  # restore checkpoint weights

            for layer in layer_keys:
                print(layer)
                # define output tensors of interest
                batch_outputs = batch_convnet.layers[layer]

                # run whatever tensors we care about
                batch_outputs = sess.run(batch_outputs)

                if idx == 0:
                    total_outputs[layer] = batch_outputs
                else:
                    total_outputs[layer] = np.concatenate(
                        (total_outputs[layer], batch_outputs), axis=0
                    )

                print(total_outputs[layer].shape)

    # let's save the features as h5py to use for fitting in another notebook
    h5f = h5py.File(
        "../../../results/models/features/alexnet/alexnet_features_subj"
        + subjid
        + "_batch"
        + str(batch_num)
        + ".h5",
        "w",
    )
    for k, v in total_outputs.items():
        h5f.create_dataset(str(k), data=v)


if __name__ == "__main__":
    # Parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjid", type=str)
    parser.add_argument("--batch_num", type=int)
    parser.add_argument("--total_batches", type=int, default=5)
    ARGS, _ = parser.parse_known_args()

    main(
        ARGS.subjid,
        ARGS.batch_num,
        ARGS.total_batches,
    )
    
