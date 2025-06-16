#!/bin/bash

set -e  # Exit on error

echo "Installing datalad and git-annex..."
pip install datalad
conda install -y -c conda-forge git-annex


echo "Initializing git-annex..."
git annex init || true  # Skip error if already initialized

echo "Installing Narratives dataset..."
datalad install -r ///labs/hasson/narratives


cd narratives

echo "Getting stimuli/ (includes wav files)..."
datalad get stimuli/

echo "Installing and getting stimuli/transcripts/"
datalad install stimuli/transcripts
datalad get stimuli/transcripts/

echo "Installing and getting gentle alignments..."
cd stimuli
datalad install gentle
cd gentle
datalad get .  # Get all align.csv files

cd ../../  # back to narratives root

task="milkyway"
echo "Downloading fMRI files for task: $task"

datalad get /home/iustin/predictive-coding-for-fMRI-LM-align-speech/narratives/sub-*/func/sub-*_task-${task}_events.tsv
datalad get /home/iustin/predictive-coding-for-fMRI-LM-align-speech/narratives/derivatives/afni-smooth/tpl-fsaverage6/tpl-fsaverage6_hemi-*_desc-cortex_mask.gii
datalad get /home/iustin/predictive-coding-for-fMRI-LM-align-speech/narratives/derivatives/afni-nosmooth/sub-*/func/sub-*_task-${task}_space-fsaverage6_hemi-*_desc-clean.func.gii

task="pieman"

datalad get /home/iustin/predictive-coding-for-fMRI-LM-align-speech/narratives/sub-*/func/sub-*_task-$task\_events.tsv
datalad get /home/iustin/predictive-coding-for-fMRI-LM-align-speech/narratives/derivatives/afni-smooth/tpl-fsaverage6/tpl-fsaverage6_hemi-*_desc-cortex_mask.gii
datalad get /home/iustin/predictive-coding-for-fMRI-LM-align-speech/narratives/derivatives/afni-nosmooth/sub-*/func/sub-*_task-$task\_space-fsaverage6_hemi-*_desc-clean.func.gii

cd ~/predictive-coding-for-fMRI-LM-align-speech

# Get all BOLDs for all tasks
datalad get narratives/derivatives/afni-nosmooth/sub-*/func/sub-*_task-*_*desc-clean_bold.nii.gz

# Get all task-level MNI brain masks
datalad get narratives/derivatives/afni-smooth/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-*_desc-brain_mask.nii.gz

cd ~/predictive-coding-for-fMRI-LM-align-speech/narratives

for task in milkyway lucy pieman; do
  for hemi in L R; do
    # Get fsaverage6 surface BOLD files
    find derivatives/afni-nosmooth/ -type f -name "*_task-${task}_space-fsaverage6_hemi-${hemi}_desc-clean.func.gii" | while read -r filepath; do
      datalad get "$filepath"
    done
  done

  # Get events.tsv files with any variant of task name
  find sub-*/func/ -type f -name "*_task-${task}*" -name "*_events.tsv" | while read -r ev_file; do
    datalad get "$ev_file"
  done

# brain parcellation files
datalad get derivatives/freesurfer/fsaverage6/label/rh.aparc.a2009s.annot
datalad get derivatives/freesurfer/fsaverage6/label/lh.aparc.a2009s.annot

echo "All downloads completed successfully."


