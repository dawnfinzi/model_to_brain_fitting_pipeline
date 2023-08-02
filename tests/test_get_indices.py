import unittest

import sys

sys.path.append("..")

from constants import *
from paths import *

sys.path.append(UTILS_PATH)

from fit_utils import get_indices, mapping

import pandas as pd
import numpy as np
import scipy.io

# Older version of get_indices, verified to work as intended for those subjects who finished the experiment (01, 02, 05, 07)
def get_indices_v0(subj):

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
    beta_order_in_73Kids = (
        all_ids[order["allixs"][0] - 1] - 1
    )  # -1 to convert from matlab to python indexing

    # shared (i.e. validation) IDS (but include all potential shared reps for the subj, not min across subjs)
    sharedix = expdesign["sharedix"][0]
    validation_mask = np.isin(rep_vals, sharedix)

    return beta_order_in_73Kids, validation_mask

class TestFeatureExtractor(unittest.TestCase):

    def testNewGetIndicesEquivalent(self):
        subj = "02"
        beta_old, _ = get_indices_v0(subj)
        beta_new, _, _ = get_indices(subj)
        
        self.assertTrue(np.all(beta_old == beta_new))


if __name__ == "__main__":
    unittest.main()