#!/usr/bin/env bash
# Input python command to be submitted as a job

#SBATCH --output=logs/class_LOO-%j.out
#SBATCH --job-name class_LOO
#SBATCH --partition=psych_day,psych_scavenge,day,scavenge_all,week,psych_gpu,gpu,scavenge,psych_week
#SBATCH --time=1:00:00 #20:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=kp578

# Set up the environment
# module load FSL
# module load Python/Anaconda3
# module load FreeSurfer/6.0.0
# module load BXH_XCEDE_TOOLS
# module load brainiak
# module load nilearn
cd /gpfs/milgram/project/turk-browne/projects/rt-cloud ; module load AFNI ; module load FSL ; source /gpfs/milgram/apps/hpc.rhel7/software/FSL/6.0.3-centos7_64/etc/fslconf/fsl.sh ; module load dcm2niix ; . /gpfs/milgram/apps/hpc.rhel7/software/Python/Anaconda3/etc/profile.d/conda.sh ; conda activate /gpfs/milgram/project/turk-browne/kp578/conda_envs/rtSynth_rt

echo python -u /gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/expScripts/recognition/class_LOO.py $1 $SLURM_ARRAY_TASK_ID
python -u /gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/expScripts/recognition/class_LOO.py $1 $SLURM_ARRAY_TASK_ID


