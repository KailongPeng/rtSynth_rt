#!/usr/bin/env bash
#SBATCH --output=logs/GM_modelTrain-%j.out
#SBATCH -p day
#SBATCH -t 24:00:00
#SBATCH --mem 20GB
#SBATCH -n 1
module load AFNI
module load FreeSurfer/6.0.0
module load FSL
. ${FSLDIR}/etc/fslconf/fsl.sh
set -e

code_dir=/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/expScripts/recognition/recognitionDataAnalysis/
raw_dir=/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/expScripts/recognition/recognitionDataAnalysis/raw/
cd ${code_dir}

sub=$1 # rtSynth_sub001 rtSynth_sub001_ses5 rtSynth_sub001 rtSynth_sub002_ses1 rtSynth_sub005_ses1
scan_asTemplate=$2

python -u -c "from GM_modelTrain_functions import _split ; _split('${sub}')"
source ${code_dir}${sub}_subjectName.txt # so you have subjectName and ses
echo subjactName=${subjectName} ses=${ses}


if [ ! -d "${raw_dir}${sub}" ]; then
    # 下载dcm数据。并且在raw_dir 里面产生{sub}_run_name.txt 用于储存每一个run分别对应什么。举例来说比如 尾数为8的代表T1 数据。
    bash ${code_dir}fetchXNAT.sh ${sub}
fi

# 通过对{sub}_run_name.txt的处理获得第二个 ABCD_T1w_MPR_vNav   usable 前面的数字
python -u -c "from GM_modelTrain_functions import find_ABCD_T1w_MPR_vNav; find_ABCD_T1w_MPR_vNav('$sub')"
source ${raw_dir}${sub}_ABCD_T1w_MPR_vNav.txt # 加载 T1_ID 由find_ABCD_T1w_MPR_vNav 产生
echo T1_ID=${T1_ID} T2_ID=${T2_ID}

# 等到zip file完成
python -u -c "from GM_modelTrain_functions import wait; wait('${raw_dir}${sub}.zip')"

if [ ! -d "${raw_dir}${sub}" ]; then
    # unzip
    cd ${raw_dir}
    unzip ${sub}.zip

    # 把dcm变成nii
    cd ${code_dir}
    bash ${code_dir}change2nifti.sh ${sub}
fi

# 根据找到的第二个T1 图像，移动到subject folder里面对应的ses的anat folder。
python -u -c "from GM_modelTrain_functions import find_T1_in_niiFolder ; find_T1_in_niiFolder(${T1_ID},${T2_ID},'${sub}')"

if [ ! -f "${anatPath}done_${subjectName}.txt" ]; then
    # 运行freesurfer
    cd /gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/expScripts/recognition/
    echo sbatch reconsurf.sh ${subjectName}
    sbatch reconsurf.sh ${subjectName}

    # 等待 freesurfer 完成
    anatPath=/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/subjects/${subjectName}/ses1/anat/
    cd ${code_dir}
    echo python -u -c "from GM_modelTrain_functions import wait; wait('${anatPath}done_${subjectName}.txt')"
    python -u -c "from GM_modelTrain_functions import wait; wait('${anatPath}done_${subjectName}.txt')"
fi

if [ ! -f "${anatPath}SUMAdone_${subjectName}.txt" ]; then
    # SUMA_Make_Spec_FS.sh
    cd /gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/expScripts/recognition/
    echo sbatch SUMA_Make_Spec_FS.sh ${subjectName}
    sbatch SUMA_Make_Spec_FS.sh ${subjectName}

    # 等待 SUMA_Make_Spec_FS 完成
    cd ${code_dir}
    echo python -u -c "from GM_modelTrain_functions import wait; wait('${anatPath}SUMAdone_${subjectName}.txt')"
    python -u -c "from GM_modelTrain_functions import wait; wait('${anatPath}SUMAdone_${subjectName}.txt')"
fi


if [ ! -f "/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/subjects/${subjectName}/ses${ses}/anat/gm_func.nii.gz" ]; then
    # 获得mask
    cd /gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/expScripts/recognition/
    echo sbatch makeGreyMatterMask.sh ${subjectName} ${scan_asTemplate}
    sbatch makeGreyMatterMask.sh ${subjectName} ${scan_asTemplate}

    # 产生的mask 类似 /gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/subjects/sub001/ses1/anat/gm_func.nii.gz
    cd ${code_dir}
    echo python -u -c "from GM_modelTrain_functions import wait; wait('/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/subjects/${subjectName}/ses${ses}/anat/gm_func.nii.gz')"
    python -u -c "from GM_modelTrain_functions import wait; wait('/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/subjects/${subjectName}/ses${ses}/anat/gm_func.nii.gz')"
fi

# 下一步是 greedy 以及 训练模型
cd /gpfs/milgram/project/turk-browne/projects/rt-cloud/

sbatch projects/rtSynth_rt/expScripts/recognition/8runRecgnitionModelTraining.sh ${subjectName}.ses${ses}.toml ${scan_asTemplate} 0 1 _ _ # config， scan_asTemplate， skipses1Greedy， skipPre， forceGreedy， tmp_folder

for leaveOutRun in {1..8} ； do
    sleep 3
    echo sbatch projects/rtSynth_rt/expScripts/recognition/ses1_LOO_Greedy_and_trainTest.sh ${subjectName}.ses1.toml ${leaveOutRun} _ #_ means tmpFolder is None
    sbatch projects/rtSynth_rt/expScripts/recognition/ses1_LOO_Greedy_and_trainTest.sh ${subjectName}.ses1.toml ${leaveOutRun} _ #_ means tmpFolder is None
done

# echo python -u projects/rtSynth_rt/expScripts/recognition/8runRecgnitionModelTraining.py -c ${subjectName}.ses${ses}.toml --scan_asTemplate ${scan_asTemplate}
# python -u projects/rtSynth_rt/expScripts/recognition/8runRecgnitionModelTraining.py --config ${subjectName}.ses${ses}.toml --scan_asTemplate ${scan_asTemplate} --skipPre

# forceGreedy=_
# tmp_folder=_
# skipses1Greedy=0
# skipPre=1

# SLURM_JOBID=18115917
# python -u projects/rtSynth_rt/expScripts/recognition/8runRecgnitionModelTraining.py --config ${subjectName}.ses${ses}.toml --scan_asTemplate ${scan_asTemplate} --forceGreedy ${forceGreedy} --tmp_folder ${tmp_folder} --skipses1Greedy ${skipses1Greedy} --jobID ${SLURM_JOBID} --skipPre ${skipPre}

# python -u projects/rtSynth_rt/expScripts/recognition/8runRecgnitionModelTraining.py --config ${config} --scan_asTemplate ${scan_asTemplate}                     --forceGreedy ${forceGreedy} --tmp_folder ${tmp_folder} --skipses1Greedy ${skipses1Greedy} --jobID ${SLURM_JOBID} --skipPre ${skipPre}