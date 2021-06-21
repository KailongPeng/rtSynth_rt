#!/usr/bin/env bash
# Input python command to be submitted as a job
#SBATCH --output=logs/FS-%j.out
#SBATCH --job-name FS
#SBATCH --partition=psych_day,psych_scavenge,psych_week,day,scavenge_all,week
#SBATCH --time=24:00:00
#SBATCH --mem=10000

# module load FSL/5.0.9
module load FreeSurfer/6.0.0
#module load BXH_XCEDE_TOOLS
#module load nilearn
# module load AFNI

subject=$1 # sub001
# source activate /gpfs/milgram/project/turk-browne/users/kp578/CONDA/rtcloud
cd /gpfs/milgram/project/turk-browne/projects/rt-cloud ; module load AFNI ; module load FSL ; source /gpfs/milgram/apps/hpc.rhel7/software/FSL/6.0.3-centos7_64/etc/fslconf/fsl.sh ; module load dcm2niix ; . /gpfs/milgram/apps/hpc.rhel7/software/Python/Anaconda3/etc/profile.d/conda.sh ; conda activate /gpfs/milgram/project/turk-browne/kp578/conda_envs/rtSynth_rt



anatPath=/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/subjects/${subject}/ses1/anat/
subjectFolder=${anatPath}freesurfer/
mkdir -p ${subjectFolder}

# process steps 1-5 
# recon-all -i ${anatPath}T1.nii -autorecon1 -notal-check -subjid ${subject} -sd ${subjectFolder};
recon-all -i ${anatPath}T1.nii -T2 ${anatPath}T2.nii -T2pial -autorecon1 -notal-check -subjid ${subject} -sd ${subjectFolder};

# process steps 6-23
recon-all -autorecon2 -subjid ${subject} -sd ${subjectFolder};

# process stages 24-31
recon-all -autorecon3 -subjid ${subject} -sd ${subjectFolder} -notify ${anatPath}done_${subject}.txt