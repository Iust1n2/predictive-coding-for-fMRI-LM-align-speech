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
done