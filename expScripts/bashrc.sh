# rtSynth_rt functions

# Mac
    function sshn { ssh -N -L 7777:172.29.189.${1}:6666 kp578@milgram.hpc.yale.edu ; } 
    # sshn 151

    function recognition { python projects/rtSynth_rt/expScripts/recognition/recognition.py -c ${1}.${2}.toml -r ${3} ${4} ; } 
    # recognition sub003 ses6 2 --trying

    function feedback { python projects/rtSynth_rt/expScripts/feedback/feedback.py -c ${1}.${2}.toml -s localhost:7777 --trying -r ${3} ; }
    # feedback sub003 ses6 1

    function chooseGeneratingCode { python projects/rtSynth_rt/expScripts/recognition/chooseGeneratingCode.py -c ${1}.${2}.toml ; } 
    # chooseGeneratingCode sub004 ses1

    alias cd_rt="cd /Users/kailong/Desktop/rtEnv/rt-cloud/ ; source activate rtSynth_rt"
    alias cd_rtt="cd /Users/kailong/Desktop/rtEnv/rt-cloud/projects/rtSynth_rt/"


# Milgram
    function pinterface { bash scripts/run-projectInterface.sh -ip 172.29.189.${1} --subjectRemote -p rtSynth_rt -c /gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/conf/${2}.${3}.toml ${4} ; } 
    # pinterface 151 sub004 ses2
    # pinterface 151 sub004 ses2 --test

    function chooseGeneratingCode { python projects/rtSynth_rt/expScripts/recognition/chooseGeneratingCode.py -c ${1}.${2}.toml ; } 
    # chooseGeneratingCode sub004 ses1

    function 2runrecognitionDataAnalysis { python projects/rtSynth_rt/expScripts/recognition/2runrecognitionDataAnalysis.py -c  ${1}.${2}.toml  --scan_asTemplate ${3} ; } 
    # 2runrecognitionDataAnalysis sub004 ses2 1

    alias cd_rt="cd /gpfs/milgram/project/turk-browne/projects/rt-cloud ; module load AFNI ; module load FSL ; source /gpfs/milgram/apps/hpc.rhel7/software/FSL/6.0.3-centos7_64/etc/fslconf/fsl.sh ; module load dcm2niix ; . /gpfs/milgram/apps/hpc.rhel7/software/Python/Anaconda3/etc/profile.d/conda.sh ; conda activate /gpfs/milgram/project/turk-browne/kp578/conda_envs/rtSynth_rt"
    alias cd_rtt="cd /gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/"

# Linux
    function recognition { python projects/rtSynth_rt/expScripts/recognition/recognition.py -c ${1}.${2}.toml -r ${3} ${4} ; } 
    # recognition sub003 ses6 2

    function feedback { python projects/rtSynth_rt/expScripts/feedback/feedback.py -c ${1}.${2}.toml -s 172.29.189.${3}:6666 -r ${4} ; }
    # feedback sub003 ses6 151 1

    alias cd_rt="cd /home/watts/Desktop/ntblab/kailong/rt-cloud/ ; conda activate rtSynth_rt"
    alias cd_rtt="cd /home/watts/Desktop/ntblab/kailong/rt-cloud/projects/rtSynth_rt/ ; conda activate rtSynth_rt"