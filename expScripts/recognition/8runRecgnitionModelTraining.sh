#!/usr/bin/env bash
#SBATCH --output=logs/8runRecgnitionModelTraining-%j.out
##SBATCH -p day
##SBATCH -t 24:00:00
#SBATCH --partition=psych_day,psych_scavenge,psych_week,day,scavenge_all,week
#SBATCH --time=6:00:00 #20:00:00
#SBATCH --mem 4GB
#SBATCH -n 1

# sbatch projects/rtSynth_rt/expScripts/recognition/8runRecgnitionModelTraining.sh sub003.ses2.toml 1 1 1 compare_forceGreedy tmp__folder_2021-06-14-16-58-20/
# config， scan_asTemplate， skipses1Greedy， skipPre， forceGreedy， tmp_folder

# 1:sub003.ses5.toml 
# 2:1 
# 3:1
# 4:1
# 5:compare_forceGreedy
# 6:tmp__folder_2021-06-14-16-58-20/

cd /gpfs/milgram/project/turk-browne/projects/rt-cloud ; module load AFNI ; module load FSL ; source /gpfs/milgram/apps/hpc.rhel7/software/FSL/6.0.3-centos7_64/etc/fslconf/fsl.sh ; module load dcm2niix ; . /gpfs/milgram/apps/hpc.rhel7/software/Python/Anaconda3/etc/profile.d/conda.sh ; conda activate /gpfs/milgram/project/turk-browne/kp578/conda_envs/rtSynth_rt

config=$1 # sub002.ses5.toml
scan_asTemplate=$2 # 1 or other number
skipses1Greedy=$3 # 1 is skipping 0 is not skipping
skipPre=$4 # 1 is skipping 0 is not skipping
forceGreedy=$5 # can be forceGreedy or compare_forceGreedy, if normal use _
tmp_folder=$6 # can be _ or the actual folder


python -u projects/rtSynth_rt/expScripts/recognition/8runRecgnitionModelTraining.py --config ${config} --scan_asTemplate ${scan_asTemplate} --forceGreedy ${forceGreedy} --tmp_folder ${tmp_folder} --skipses1Greedy ${skipses1Greedy} --jobID ${SLURM_JOBID} --skipPre ${skipPre}
