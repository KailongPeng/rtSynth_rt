
import os
import sys
sys.path.append('/gpfs/milgram/project/turk-browne/projects/rt-cloud/')
sys.path.append('/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/')
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
from shutil import copyfile
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

argParser = argparse.ArgumentParser()
argParser.add_argument('--config', '-c', default='sub002.ses4.toml', type=str, help='experiment file (.json or .toml)')
# argParser.add_argument('--skipPre', '-s', default=0, type=int, help='skip preprocess or not')
# argParser.add_argument('--skipGreedy', '-g', default=0, type=int, help='skip greedy or not')
# argParser.add_argument('--testRun', '-t', default=None, type=int, help='testRun, can be [None,1,2,3,4,5,6,7,8]')
argParser.add_argument('--scan', '-s', default=1, type=int, help="which scan to simulate")

args = argParser.parse_args()
from cfg_loading import mkdir,cfg_loading
# config="sub001.ses2.toml"
cfg = cfg_loading(args.config)

# tmp_folder='/gpfs/milgram/scratch60/turk-browne/kp578/dicom_folder/'
# shutil.rmtree("/tmp/dicom_folder/")
shutil.rmtree(cfg.dicom_dir)
mkdir(cfg.dicom_dir)

_t = input('Start now? Type anything \n')
print(_t)
startTime = time.time()
curr_TR=1
while True:
    currTime=time.time()
    if currTime - startTime > 2-0.002:
        curr_dicom=f"{cfg.old_dicom_dir}001_{str(args.scan).zfill(6)}_{str(curr_TR).zfill(6)}.dcm"
        copyfile(curr_dicom,f"{cfg.dicom_dir}001_{str(args.scan).zfill(6)}_{str(curr_TR).zfill(6)}.dcm")
        print(f"curr_TR={curr_TR}")
        curr_TR+=1
        startTime+=2
        if curr_TR>180:
            break


# for curr_TR in range(1,180+1):
#     time.sleep(2)
#     curr_dicom=f"{cfg.old_dicom_dir}001_{str(args.scan).zfill(6)}_{str(curr_TR).zfill(6)}.dcm"
#     copyfile(curr_dicom,f"{cfg.dicom_dir}001_{str(args.scan).zfill(6)}_{str(curr_TR).zfill(6)}.dcm")
#     print(f"curr_TR={curr_TR}")
