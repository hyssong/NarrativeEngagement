# NarrativeEngagement

**Song, Finn, & Rosenberg (2021) Neural signatures of attentional engagement during narratives and its consequences for event memory**

FMRI datasets used in the study
- Sherlock dataset : http://arks.princeton.edu/ark:/88435/dsp01nz8062179 (Chen et al., 2017; *Nat. Neurosci.*)
- Paranoia dataset : https://openneuro.org/datasets/ds001338/versions/1.0.0 (Finn et al., 2018; *Nat. Commun.*)

**data**
- behavior-emotion-* : positive and negative emotional arousal
- behavior-engagement-* : behavioral experiment results (1-9 ratings; N=21 for Paranoia, N=17 for Sherlock, collected at the University of Chicago)
- behavior-sensory-* : auditory envelope and global luminance
- fmri-BOLD-* : fMRI participants' cortical (114) + subcortical (8) parcellated BOLD timeseries (Yeo et al., 2015; *Neuroimage*)
- fmri-recall-* : fMRI participants' post-hoc behavioral experiment results; recall fidelity timecourse (Heusser et al., 2021; *Nat. Hum. Behav.*)
- hyperparameters : list of hyperparameters used in the study

**code**
- preprocess (*needs to run before conducting analysis)
  - slidingBeh.m : Generate group-average engagement timecourse (applies HRF convolution and sliding window to relate with fMRI data)
  - slidingBeh_surr.m : Generate phase-randomized behavioral timecourses for non-parametric permutation test
  - slidingFC.m : Generate time-resolved functional connectivity matrices from BOLD timeseries, using sliding window analysis
- analysis
  - dynISC.m : Dynamic inter-subject correlation analysis, related to narrative engagement timecourse
  - within_engagement_dynPred.ipynb : Functional connectivity-based, within-dataset prediction of time-varying engagement (*needs to run before running across_engagement_dynPred.ipynb)
  - across_engagement_dynPred.ipynb : Engagement network-based, across-dataset prediction of time-varying engagement
