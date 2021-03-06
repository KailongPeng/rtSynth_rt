#  this script is meant to deal with the data of 8 recognition runs and generate models saved in corresponding folder
'''
input:
    cfg.session=ses1
    cfg.modelFolder=f"{cfg.subjects_dir}/{cfg.subjectName}/{cfg.session}_recognition/clf/"
    cfg.dataFolder=f"{cfg.subjects_dir}/{cfg.subjectName}/{cfg.session}_recognition/"
output:
    models in cfg.modelFolder
'''


import os
import sys
sys.path.append('/gpfs/milgram/project/turk-browne/projects/rt-cloud/')
sys.path.append('/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/')
sys.path.append('/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/expScripts/recognition/')

import argparse
import numpy as np
import nibabel as nib
import scipy.io as sio
import subprocess
from scipy.stats import zscore
from nibabel.nicom import dicomreaders
import pydicom as dicom  # type: ignore
import time
from glob import glob
import shutil
from nilearn.image import new_img_like
import joblib
import rtCommon.utils as utils
from rtCommon.utils import loadConfigFile
import pickle5 as pickle
def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
# from rtCommon.fileClient import FileInterface
# import rtCommon.projectUtils as projUtils
# from rtCommon.imageHandling import readRetryDicomFromFileInterface, getDicomFileName, convertDicomImgToNifti


argParser = argparse.ArgumentParser()
argParser.add_argument('--config', '-c', default='sub005.ses1.toml', type=str, help='experiment file (.json or .toml)')
argParser.add_argument('--skipPre', '-s', default=0, type=int, help='skip preprocess or not')
argParser.add_argument('--skipses1Greedy', '-g', default=0, type=int, help='skip greedy or not') #1 is skip 0 is not skip
argParser.add_argument('--forceGreedy', default='_', type=str, help='whether to force Greedy search in current session, can be compare_forceGreedy , forceGreedy or _')
argParser.add_argument('--testRun', '-t', default=None, type=int, help='testRun, can be [None,1,2,3,4,5,6,7,8]')
argParser.add_argument('--scan_asTemplate', '-a', default=1, type=int, help="which scan's middle dicom as Template?")
argParser.add_argument('--preprocessOnly', default=0, type=int, help='whether to only do preprocess and skip everything else')
argParser.add_argument('--tmp_folder', default='_' , type=str, help='tmp_folder')
argParser.add_argument('--jobID', default='' , type=str, help='jobID')


args = argParser.parse_args()
from cfg_loading import mkdir,cfg_loading
# config="sub001.ses2.toml"
cfg = cfg_loading(args.config)
cfg.jobID=args.jobID

from recognition_dataAnalysisFunctions import recognition_preprocess,minimalClass,behaviorDataLoading,greedyMask,normalize #,classifierEvidence
def wait(waitfor, delay=1):
    while not os.path.exists(waitfor):
        time.sleep(delay)
        print('waiting for {}'.format(waitfor))

recordingTxt=f"{cfg.subjects_dir}{cfg.subjectName}/ses{cfg.session}/recognition/recording.txt" # None
try:
    tmp_folder=args.tmp_folder
except:
    tmp_folder='_' # this is used when I don't want to give any tmp_folder

print(f"tmp_folder={tmp_folder}")

if args.preprocessOnly:
    recognition_preprocess(cfg,args.scan_asTemplate)
else:
    '''
    convert all dicom files into nii files in the temp dir. 
    find the middle volume of the run1 as the template volume
    align every other functional volume with templateFunctionalVolume (3dvolreg)
    '''
    if not args.skipPre:
        recognition_preprocess(cfg,args.scan_asTemplate) #somehow this cannot be run in jupyter


    '''
    run the mask selection
        make ROIs
            make-schaefer-rois.sh
        starting from 31 megaROIs use greedyMask to find best ROI for the current subject
    '''
    # make ROIs
    if cfg.session==1:
        if not os.path.exists(f"{cfg.recognition_dir}mask/GMschaefer_300.nii.gz"):
            print(f"running sbatch {cfg.recognition_expScripts_dir}make-schaefer-rois.sh {cfg.subjectName} {cfg.recognition_dir}")
            subprocess.Popen(f"sbatch {cfg.recognition_expScripts_dir}make-schaefer-rois.sh {cfg.subjectName} {cfg.recognition_dir}",shell=True)
            wait(f"{cfg.recognition_dir}mask/GMschaefer_300.nii.gz")

        # when this is the first session, you need to select the chosenMask
        # python expScripts/recognition/greedyMask.py
        if not args.skipses1Greedy:
            print("running greedyMask")
            recordingTxt=greedyMask(cfg,forceGreedy=args.forceGreedy,tmp_folder=tmp_folder)

    
    if args.forceGreedy=='forceGreedy':
        print("force running greedyMask")
        # cfg.chosenMask=f"{cfg.subjects_dir}{cfg.subjectName}/ses{cfg.session}/recognition/chosenMask.npy"
        recordingTxt=greedyMask(cfg,forceGreedy=args.forceGreedy,tmp_folder=tmp_folder)
    elif args.forceGreedy=='compare_forceGreedy':
        print("compare_forceGreedy, skipping greedyMask")
        pass
    elif args.forceGreedy=='_':
        print("forceGreedy== _ , skipping greedyMask")
        pass
    else:
        raise Exception("args.forceGreedy is wrong, should be compare_forceGreedy , forceGreedy or _")

    # train the classifiers
    # accs = minimalClass(cfg)
    accs, cfg = minimalClass(cfg,testRun=args.testRun,recordingTxt=recordingTxt, forceGreedy=args.forceGreedy)

    print("\n\n")
    print(f"minimalClass accs={accs}")
    save_obj(accs,f"{cfg.recognition_dir}minimalClass_accs")


# '''
# run the mask selection
#     make ROIs
#         make-wang-rois.sh
#         make-schaefer-rois.sh
#     train classifiers on each ROI
#     summarize classification accuracy and select best mask
# '''
# # make ROIs
#     # make-wang-rois.sh
# subprocess.Popen(f"sbatch {cfg.recognition_expScripts_dir}make-wang-rois.sh {cfg.subjectName} {cfg.recognition_dir}",shell=True)
#     # make-schaefer-rois.sh
# subprocess.Popen(f"sbatch {cfg.recognition_expScripts_dir}make-schaefer-rois.sh {cfg.subjectName} {cfg.recognition_dir}",shell=True)
# wait(f"{cfg.recognition_dir}mask/schaefer_300.nii.gz")
# wait(f"{cfg.recognition_dir}mask/wang_roi25_lh.nii.gz")

# # train classifiers on each ROI 
# subprocess.Popen(f"sbatch {cfg.recognition_expScripts_dir}batchRegions.sh {config}",shell=True)

# # summarize classification accuracy and select best mask
# subprocess.Popen(f"bash {cfg.recognition_expScripts_dir}runAggregate.sh {config}",shell=True)
# # bash /gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/expScripts/recognition/runAggregate.sh sub001.ses1.toml

# # select the mask with the best performance as cfg.chosenMask = {cfg.recognition_dir}chosenMask.nii.gz
# # and also save this mask in all 

# '''
# load preprocessed and aligned behavior and brain data 
# select data with the wanted pattern like AB AC AD BC BD CD 
# train correspondng classifier and save the classifier performance and the classifiers themselves.
# '''
# minimalClass(cfg)


