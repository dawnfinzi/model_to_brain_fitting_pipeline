import sys

from torch._C import PyTorchFileWriter

sys.path.append("..")
from constants import *

from helper import write_boilerplate

script = "${OAK}/biac2/kgs/projects/Dawn/NSD/code/fit_pipeline/scripts/overall_fitting_wrapper.py"

# set variables

subject_names = ["01", "02", "03", "04", "05", "06", "07", "08"]  # , "01"]
hemis = ["rh", "lh"]  # , "lh"]
rois = ["streams_shrink10"]
model_names = ["spacetorch"]
mapping_names = ["Ridge"]  # , "Ridge"]
CV_opts = [1]
subsamp_opts = [2]  # , 1]
pretrained = [1]
spatial_weight = 0.25

for subject_name in subject_names:
    for h in hemis:
        for roi in rois:
            for model_name in model_names:
                for mapping_name in mapping_names:
                    for pt in pretrained:
                        for CV in CV_opts:
                            if CV == 1:
                                time = "36:00:00"  # increased time limit
                            else:
                                time = "24:00:00"  # standard

                            mem = "256000"

                            for subsamp in subsamp_opts:

                                description = (
                                    f"{subject_name}_{spatial_weight}_{h}_{mapping_name}"
                                    f"_CV{CV}_subsamp{subsamp}_{model_name}_{roi}_{pt}"
                                )

                                sbatch_filename = f"fitting/{description}.sbatch"
                                f = open(sbatch_filename, "w")

                                write_boilerplate(
                                    f, jobname=description, time=time, mem=mem
                                )

                                f.write(
                                    (
                                        "\n"
                                        "source ~/fitting_venv/bin/activate"
                                        "\n"
                                        f"srun python {script} "
                                        f"--subj '{subject_name}' "
                                        f"--hemi '{h}' "
                                        f"--roi '{roi}' "
                                        f"--model_name '{model_name}' "
                                        f"--subsample {subsamp} "
                                        f"--mapping_func '{mapping_name}' "
                                        f"--CV {CV} "
                                        f"--pretrained {pt} "
                                        f"--reduce_temporal_dims 0 "
                                        f"--spatial_weight {spatial_weight} "
                                    )
                                )
