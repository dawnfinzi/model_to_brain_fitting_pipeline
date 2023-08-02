#!/bin/bash

declare -a ROI_NAME=( "streams_shrink10" )

declare -a SUBJECTS=( "01" "02" "03" "04" "05" "06" "07" "08" )
declare -a HEMIS=( "lh" "rh" )

declare -a MODELS=( "slowfast_full" )

# MAP FUNCS
declare -a MAP_FUNCS=( "Ridge" )

# CV
declare -a CV=( "1" )
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

                            filename="layer_${subject}_${hemi}_${mapping}_CV${c}_subsamp${s}_${model}_${ROI_NAME}_${p}.sbatch"

                            sbatch $filename
                            echo "layer ${subject}, ${hemi}, ${mapping}, ${c}, ${s}, ${model}, ${p}"
                        done
                    done
                done
            done
        done
    done
done
