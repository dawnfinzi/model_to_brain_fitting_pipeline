#!/bin/bash

declare -a SUBJECTS=( "01" "02" "03" "04" "05" "06" "07" "08" )
declare -a HEMIS=( "lh" )

declare -a TARGET_ROI_NAME=( "streams_shrink10" )
declare -a SOURCE_ROI_NAME=( "streams_shrink10" )

declare -a N_VOXELS=( "10000" )
declare -a NUM_SPLITS=( "10" )

for subject in ${SUBJECTS[@]};
do
    for hemi in ${HEMIS[@]};
    do
        for target in ${TARGET_ROI_NAME[@]};
        do
            for source in ${SOURCE_ROI_NAME[@]};
            do
                for n in ${N_VOXELS[@]};
                do
                    for splits in ${NUM_SPLITS[@]};
                    do

                        filename="${subject}_${hemi}_Ridge1_target_${target}_source_${source}_${n}voxels_${splits}splits_1_subsamp1.sbatch"

                        sbatch $filename
                        echo "${subject}, ${hemi}, ${target}, ${source}, ${n}, ${splits}"
                    done
                done
            done
        done
    done
done
