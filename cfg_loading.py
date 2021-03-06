import rtCommon.utils as utils
import os
print(f"conda env={os.environ['CONDA_DEFAULT_ENV']}") 
# source activate /gpfs/milgram/project/turk-browne/users/kp578/CONDA/rtcloud

# sys.path.append('/gpfs/milgram/project/turk-browne/users/kp578/realtime/rt-cloud/')
def mkdir(folder):
    import numpy as np
    if not os.path.isdir(folder):
        os.mkdir(folder)
    np.save(f"{folder}/folderKeeper.npy",1)

def cfg_loading(toml=''):
    def findDir(path):
        from glob import glob
        # _path = glob(path)[0]+'/'
        _path = glob(path)
        if len(_path)==0: # if the dir is not found. get rid of the "*" and return
            _path=path.split("*")
            _path=''.join(_path)
        else:
            _path = _path[0]+'/'
        return _path

    # toml="pilot_sub001.ses1.toml"
    # cfg = utils.loadConfigFile(f"/gpfs/milgram/project/turk-browne/users/kp578/realtime/rt-cloud/projects/rtSynth_rt/conf/{toml}")
    if len(toml.split("/"))==1:
        if 'watts' in os.getcwd():
            toml=f"/home/watts/Desktop/ntblab/kailong/rt-cloud/projects/rtSynth_rt/conf/{toml}"
        elif 'kailong' in os.getcwd():
            toml=f"/Users/kailong/Desktop/rtEnv/rt-cloud/projects/rtSynth_rt/conf/{toml}"
        elif 'milgram' in os.getcwd():
            toml=f"/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/conf/{toml}"
        else:
            raise Exception('path error')
    else:
        toml=toml
    
    if 'watts' in os.getcwd():
        cfg = utils.loadConfigFile(toml)
        cfg.projectDir="/home/watts/Desktop/ntblab/kailong/rt-cloud/projects/rtSynth_rt/"
    elif 'kailong' in os.getcwd():
        cfg = utils.loadConfigFile(toml)
        cfg.projectDir="/Users/kailong/Desktop/rtEnv/rt-cloud/projects/rtSynth_rt/"
    elif 'milgram' in os.getcwd():
        cfg = utils.loadConfigFile(toml)
        cfg.projectDir="/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/"
    else: 
        raise Exception('path error')
    
    cfg.orderFolder=f'{cfg.projectDir}expScripts/recognition/orders/'
    cfg.subjects_dir=f'{cfg.projectDir}subjects/'
    cfg.dicom_folder="/gpfs/milgram/project/realtime/DICOM/"
    cfg.recognition_expScripts_dir = f"{cfg.projectDir}expScripts/recognition/"
    cfg.feedback_expScripts_dir = f"{cfg.projectDir}expScripts/feedback/"

    

    cfg.preDay_dicom_dir  = findDir(f"{cfg.dicom_folder}{cfg.preDay_YYYYMMDD}.{cfg.LASTNAME}*.{cfg.LASTNAME}*/")  #e.g. /gpfs/milgram/project/realtime/DICOM/20201009.rtSynth_pilot001.rtSynth_pilot001/  # cfg.preDay_YYYYMMDD is "0" when there is no previous day    
    if cfg.trying is True:
        # cfg.dicom_dir="/tmp/dicom_folder/"
        cfg.dicom_dir='/gpfs/milgram/scratch60/turk-browne/kp578/dicom_folder/'
        cfg.old_dicom_dir     = findDir(f"{cfg.dicom_folder}{cfg.YYYYMMDD}.{cfg.LASTNAME}*.{cfg.LASTNAME}*/")  # YYYYMMDD.$LASTNAME.$PATIENTID  e.g. /gpfs/milgram/project/realtime/DICOM/20201019.rtSynth_pilot001_2.rtSynth_pilot001_2/ inside which is like 001_000003_000067.dcm    
        cfg.TR=2
        cfg.tmp_folder="/gpfs/milgram/scratch60/turk-browne/kp578/rtcloud_rt/"
    else:
        cfg.dicom_dir     = findDir(f"{cfg.dicom_folder}{cfg.YYYYMMDD}.{cfg.LASTNAME}*.{cfg.LASTNAME}*/")  # YYYYMMDD.$LASTNAME.$PATIENTID  e.g. /gpfs/milgram/project/realtime/DICOM/20201019.rtSynth_pilot001_2.rtSynth_pilot001_2/ inside which is like 001_000003_000067.dcm    
        cfg.TR=2 #2 #?????????TR???2s
        cfg.tmp_folder="/tmp/" # for speeding up, use the storage on the local memory unit
    cfg.dicomDir          = cfg.dicom_dir
    cfg.recognition_dir   = f"{cfg.subjects_dir}{cfg.subjectName}/ses{cfg.session}/recognition/"
    cfg.feedback_dir      = f"{cfg.subjects_dir}{cfg.subjectName}/ses{cfg.session}/feedback/"
    cfg.usingModel_dir    = f"{cfg.subjects_dir}{cfg.subjectName}/ses{cfg.session-1}/recognition/clf/"
    cfg.trainingModel_dir = f"{cfg.subjects_dir}{cfg.subjectName}/ses{cfg.session}/recognition/clf/"

    cfg.chosenMask = f"{cfg.subjects_dir}{cfg.subjectName}/ses1/recognition/chosenMask.npy"
    cfg.chosenMask_training=f"{cfg.subjects_dir}{cfg.subjectName}/ses{cfg.session}/recognition/chosenMask.npy"
    cfg.chosenMask_using=f"{cfg.subjects_dir}{cfg.subjectName}/ses{cfg.session-1}/recognition/chosenMask.npy"

    cfg.templateFunctionalVolume = f"{cfg.subjects_dir}{cfg.subjectName}/ses1/recognition/templateFunctionalVolume.nii" 
    cfg.templateFunctionalVolume_converted = f"{cfg.recognition_dir}/templateFunctionalVolume_converted.nii" # templateFunctionalVolume_converted is the current day run1 middle volume converted in day1 template space
    cfg.dicomNamePattern  = "001_{SCAN:06d}_{TR:06d}.dcm" # "001_0000{}_000{}.dcm"
    cfg.mask_dir          = f"{cfg.recognition_dir}mask/"
    cfg.GMINFUNC=f"{cfg.subjects_dir}{cfg.subjectName}/ses1/anat/gm_func.nii.gz"
    cfg.adaptiveThreshold=f"{cfg.subjects_dir}{cfg.subjectName}/adaptiveThreshold.csv"
    cfg.twoWayClfDict={
        "AB":['bedtable_bedchair','\nbedtable_tablebench'],
        "AC":['bedchair_bedtable','\nbedchair_chairbench'],
        "AD":['bedchair_bedbench','\nbedchair_bedtable'],
        "BC":['bedchair_chairtable','\nbedtable_bedbench'],
        "BD":['bedchair_chairbench','\nbedchair_chairtable'],
        "CD":['bedtable_tablebench','\nbedtable_tablechair'],
    }
    # prepare folder structure
    mkdir(f"{cfg.subjects_dir}{cfg.subjectName}") # mkdir subject folder
    for curr_ses in [1,5]: # keep 6 here just for sub002
        mkdir(f"{cfg.subjects_dir}{cfg.subjectName}/ses{curr_ses}/")
        mkdir(f"{cfg.subjects_dir}{cfg.subjectName}/ses{curr_ses}/catPer/")
        mkdir(f"{cfg.subjects_dir}{cfg.subjectName}/ses{curr_ses}/anat/")
        mkdir(f"{cfg.subjects_dir}{cfg.subjectName}/ses{curr_ses}/recognition/")
        mkdir(f"{cfg.subjects_dir}{cfg.subjectName}/ses{curr_ses}/recognition/clf/")
        mkdir(f"{cfg.subjects_dir}{cfg.subjectName}/ses{curr_ses}/feedback/") # keep this here just for sub002

    for curr_ses in [2,3,4]:
        mkdir(f"{cfg.subjects_dir}{cfg.subjectName}/ses{curr_ses}/")
        mkdir(f"{cfg.subjects_dir}{cfg.subjectName}/ses{curr_ses}/recognition/")
        mkdir(f"{cfg.subjects_dir}{cfg.subjectName}/ses{curr_ses}/recognition/clf/")
        mkdir(f"{cfg.subjects_dir}{cfg.subjectName}/ses{curr_ses}/feedback/")

    return cfg



# if os.path.isdir(tmp_folder):
#   shutil.rmtree(tmp_folder)
