#!/usr/bin/env bash
#SBATCH --job-name ses1_LOO_Greedy_and_trainTest
#SBATCH --output=logs/ses1_LOO_Greedy_and_trainTest-%j.out
#SBATCH --partition=psych_day,psych_scavenge,psych_week,day,scavenge_all,week
#SBATCH --time=6:00:00 #20:00:00
#SBATCH --mem 4GB
#SBATCH -n 1
cd /gpfs/milgram/project/turk-browne/projects/rt-cloud ; module load AFNI ; module load FSL ; source /gpfs/milgram/apps/hpc.rhel7/software/FSL/6.0.3-centos7_64/etc/fslconf/fsl.sh ; module load dcm2niix ; . /gpfs/milgram/apps/hpc.rhel7/software/Python/Anaconda3/etc/profile.d/conda.sh ; conda activate /gpfs/milgram/project/turk-browne/kp578/conda_envs/rtSynth_rt

toml=$1 # sub002.ses5.toml
LeaveOutRun=$2
tmp_folder=$3
echo python -u /gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/expScripts/recognition/ses1_LOO_Greedy_and_trainTest.py -c $toml --LeaveOutRun $LeaveOutRun --tmp_folder $tmp_folder --jobID ${SLURM_JOBID}
python -u /gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/expScripts/recognition/ses1_LOO_Greedy_and_trainTest.py -c $toml --LeaveOutRun $LeaveOutRun --tmp_folder $tmp_folder --jobID ${SLURM_JOBID}

