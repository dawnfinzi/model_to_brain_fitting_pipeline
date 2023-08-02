import argparse
import nibabel.freesurfer.mghformat as mgh
import h5py

import sys

sys.path.append("..")

from paths import *
from constants import ROI_NAMES

sys.path.append(UTILS_PATH)
from rsm_utils import get_ROI_data


def main(subj, hemi, roi):

    betas = h5py.File(
        BETA_PATH + "datab3nativesurface_subj" + subj + "_" + hemi + "_betas.hdf5", "r"
    )

    # get ROI data to pick which voxels to fit
    mgh_file = mgh.load(FS_PATH + "subj" + subj + "/" + hemi + "." + roi + ".mgz")
    streams = mgh_file.get_fdata()[:, 0, 0]

    for ridx, r_name in enumerate(ROI_NAMES):

        if ridx > 0:  # ignore unclassified vertices

            # subset by area
            area_betas = betas["betas"][:, streams == ridx]

            h5f = h5py.File(
                LOCALDATA_PATH
                + "processed/organized_betas_by_area/subj"
                + subj
                + "_"
                + hemi
                + "_"
                + roi
                + "_"
                + r_name
                + ".hdf5",
                "w",
            )

            h5f.create_dataset(r_name, data=area_betas)
            h5f.close()


if __name__ == "__main__":
    # Parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--subj", type=str)
    parser.add_argument("--hemi", type=str, default="rh")
    parser.add_argument("--roi", type=str, default="streams_shrink10")
    ARGS, _ = parser.parse_known_args()

    main(
        ARGS.subj,
        ARGS.hemi,
        ARGS.roi,
    )