from re import sub
import sys

sys.path.append("..")
from constants import *

from helper import write_boilerplate

script = (
    "${OAK}/biac2/kgs/projects/Dawn/NSD/code/fit_pipeline/scripts/subject2subject.py"
)

# set variables

subject_names = ["01", "02", "03", "04", "05", "06", "07", "08"]
hemis = ["rh"]
target_rois = ["streams_shrink10"]
source_rois = ["streams_shrink10"]
n_source_voxels = [None]
num_splits = [10]
mapping_func = "PLS"
CV = 0
pooled = 0
subsamp_type = 1

for subject_name in subject_names:
    for h in hemis:
        for target_roi in target_rois:
            for source_roi in source_rois:
                for n in n_source_voxels:
                    for splits in num_splits:
                        if CV == 1:
                            time = "48:00:00"  # increased time limit
                        else:
                            time = "4:00:00"  # standard

                        mem = "512000"

                        description = (
                            f"{subject_name}_{h}_{mapping_func}{CV}_target_{target_roi}"
                            f"_source_{source_roi}_{n}voxels_{splits}splits_{pooled}_subsamp{subsamp_type}"
                        )
                        sbatch_filename = f"subj2subj/{description}.sbatch"
                        f = open(sbatch_filename, "w")

                        write_boilerplate(f, jobname=description, time=time, mem=mem)

                        if n is None:
                            f.write(
                                (
                                    "\n"
                                    "source ~/fitting_venv/bin/activate"
                                    "\n"
                                    f"srun python {script} "
                                    f"--target_subj '{subject_name}' "
                                    f"--hemi '{h}' "
                                    f"--target_roi '{target_roi}' "
                                    f"--source_roi '{source_roi}' "
                                    f"--num_splits {splits} "
                                    f"--subsamp_type {subsamp_type} "
                                    f"--mapping_func '{mapping_func}' "
                                    f"--CV {CV} "
                                    f"--pooled {pooled}"
                                )
                            )
                        else:
                            f.write(
                                (
                                    "\n"
                                    "source ~/fitting_venv/bin/activate"
                                    "\n"
                                    f"srun python {script} "
                                    f"--target_subj '{subject_name}' "
                                    f"--hemi '{h}' "
                                    f"--target_roi '{target_roi}' "
                                    f"--source_roi '{source_roi}' "
                                    f"--n_source_voxels {n} "
                                    f"--num_splits {splits} "
                                    f"--subsamp_type {subsamp_type} "
                                    f"--mapping_func '{mapping_func}' "
                                    f"--CV {CV} "
                                    f"--pooled {pooled}"
                                )
                            )
