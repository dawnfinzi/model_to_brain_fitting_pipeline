import sys

from torch._C import PyTorchFileWriter

sys.path.append("..")
from constants import *

from helper import write_boilerplate_parallel

script = "${OAK}/biac2/kgs/projects/Dawn/NSD/code/fit_pipeline/scripts/overall_fitting_by_layer_wrapper.py"

# set variables

subject_names = ["01", "02", "03", "04", "05", "06", "07", "08"]  # , "01"]
hemis = ["rh", "lh"]  # , "lh"]
rois = ["streams_shrink10"]
model_names = ["slowfast_full"]
mapping_names = ["Ridge"]  # , "Ridge"]
CV_opts = [1]
subsamp_opts = [2]  # , 1]
pretrained = [1]

for subject_name in subject_names:
    for h in hemis:
        for roi in rois:
            for model_name in model_names:
                for mapping_name in mapping_names:
                    for pt in pretrained:
                        for CV in CV_opts:
                            time = "48:00:00"  # standard

                            mem = "512000"
                            # if subject_name == "01":  # big subjects
                            #    mem = "512000"

                            for subsamp in subsamp_opts:

                                description = (
                                    f"layer_{subject_name}_{h}_{mapping_name}"
                                    f"_CV{CV}_subsamp{subsamp}_{model_name}_{roi}_{pt}"
                                )

                                sbatch_filename = (
                                    f"fitting_by_layer/{description}.sbatch"
                                )
                                f = open(sbatch_filename, "w")

                                write_boilerplate_parallel(
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
                                        f"--reduce_temporal_dims 1 "
                                    )
                                )
