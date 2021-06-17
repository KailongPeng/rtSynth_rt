#!/usr/bin/env bash
#SBATCH --output=logs/8runRecgnitionModelTraining-%j.out
##SBATCH -p day
##SBATCH -t 24:00:00
#SBATCH --partition=psych_day,psych_scavenge,psych_week,day,scavenge_all,week
#SBATCH --time=6:00:00 #20:00:00
#SBATCH --mem 4GB
#SBATCH -n 1

cd /gpfs/milgram/project/turk-browne/projects/rt-cloud ; module load AFNI ; module load FSL ; source /gpfs/milgram/apps/hpc.rhel7/software/FSL/6.0.3-centos7_64/etc/fslconf/fsl.sh ; module load dcm2niix ; . /gpfs/milgram/apps/hpc.rhel7/software/Python/Anaconda3/etc/profile.d/conda.sh ; conda activate /gpfs/milgram/project/turk-browne/kp578/conda_envs/rtSynth_rt

toml=$1 # sub002.ses5.toml
scan_asTemplate=$2 # 1
forceGreedy=$3 # 1
tmp_folder=$4
skipGreedy=$5 # 1 is skipping 0 is not skipping
if [ "$forceGreedy" == "1" ]; then
    echo "forceGreedy"
    python -u projects/rtSynth_rt/expScripts/recognition/8runRecgnitionModelTraining.py -c $toml --scan_asTemplate $scan_asTemplate --skipPre --forceGreedy --tmp_folder $tmp_folder
else
    echo "not forceGreedy"
    python -u projects/rtSynth_rt/expScripts/recognition/8runRecgnitionModelTraining.py -c $toml --scan_asTemplate $scan_asTemplate --skipGreedy ${skipGreedy}
fi

