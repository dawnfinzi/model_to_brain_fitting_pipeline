#!/bin/bash

declare -a ROI_NAME=( "streams_shrink10" )

declare -a SUBJECTS=( "01" "02" "03" "04" "05" "06" "07" "08" )
declare -a HEMIS=( "rh" )

declare -a MODELS=( "alexnet_torch" "vgg16" "cornet-s" "resnet50" "resnet101" )

# MAP FUNCS
declare -a MAP_FUNCS=( "PLS" )

# CV
declare -a CV=( "0" )
# subsamp
declare -a SS=( "2" )
# pretrained
declare -a PT=( "1" )


for subject in ${SUBJECTS[@]};
do
    for hemi in ${HEMIS[@]};
    do
        for mapping in ${MAP_FUNCS[@]};
        do
            for c in ${CV[@]};
            do
                for s in ${SS[@]};
                do
                    for p in ${PT[@]};
                    do
                        for model in ${MODELS[@]};
                        do

                            filename="area_${subject}_${hemi}_${mapping}_CV${c}_subsamp${s}_${model}_${ROI_NAME}_${p}.sbatch"

                            sbatch $filename
                            echo "area ${subject}, ${hemi}, ${mapping}, ${c}, ${s}, ${model}, ${p}"
                        done
                    done
                done
            done
        done
    done
done
