import sys

from torch._C import PyTorchFileWriter

sys.path.append("..")
from constants import *

from helper import write_boilerplate_parallel

script = "${OAK}/biac2/kgs/projects/Dawn/NSD/code/fit_pipeline/scripts/fitting_by_area_wrapper.py"

# set variables

subject_names = ["01", "02", "03", "04", "05", "06", "07", "08"]
hemis = ["rh"]  # , "lh"]
rois = ["streams_shrink10"]
model_names = ["alexnet_torch", "vgg16", "cornet-s", "resnet50", "resnet101"]
mapping_names = ["PLS"]  # , "Ridge"]
CV_opts = [0]
subsamp_opts = [2]  # , 1]
pretrained = [1]

for subject_name in subject_names:
    for h in hemis:
        for roi in rois:
            for model_name in model_names:
                for mapping_name in mapping_names:
                    for pt in pretrained:
                        for CV in CV_opts:
                            time = "12:00:00"  # standard

                            mem = "256000"

                            for subsamp in subsamp_opts:

                                description = (
                                    f"area_{subject_name}_{h}_{mapping_name}"
                                    f"_CV{CV}_subsamp{subsamp}_{model_name}_{roi}_{pt}"
                                )

                                sbatch_filename = (
                                    f"fitting_by_area/{description}.sbatch"
                                )
                                f = open(sbatch_filename, "w")

                                write_boilerplate_parallel(
                                    f,
                                    jobname=description,
                                    time=time,
                                    mem=mem,
                                    n_task_ids=6,
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
                                        f"--reduce_temporal_dims 1 "
                                    )
                                )
