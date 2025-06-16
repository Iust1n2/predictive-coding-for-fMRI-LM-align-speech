#!/bin/bash
cd ~/predictive-coding-for-fMRI-LM-align-speech/narratives

# Install subdatasets recursively
datalad install -r .

# Get stimuli directory
datalad get stimuli/

# Get brain masks for each task
for task in milkyway lucy pieman; do
    datalad get "derivatives/afni-smooth/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-${task}_desc-brain_mask.nii.gz"
done

# Get afni-nosmooth BOLD data (clean MNI BOLD)
for task in milkyway lucy pieman; do
    while IFS= read -r bold_file; do
        datalad get "$bold_file"
    done < <(find derivatives/afni-nosmooth/ -type f -name "*_task-${task}_space-MNI152NLin2009cAsym_res-native_desc-clean_bold.nii.gz")
done

# OPTIONAL: Get only subject raw folders containing those tasks
for task in milkyway lucy pieman; do
    for sub in $(grep -l "${task}" sub-*/func/*events.tsv | cut -d/ -f1 | sort -u); do
        datalad get "$sub"
    done
done

