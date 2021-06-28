verbose=True
import os,time
import sys
# sys.path.append('/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/')
sys.path.append('/gpfs/milgram/project/turk-browne/projects/rt-cloud/')
sys.path.append('/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/')
sys.path.append('/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/expScripts/recognition/')
import argparse
import warnings
import numpy as np
import nibabel as nib
import scipy.io as sio
from cfg_loading import mkdir,cfg_loading
from subprocess import call
import joblib
import pandas as pd
from scipy.stats import zscore
if verbose:
    print(''
        '|||||||||||||||||||||||||||| IGNORE THIS WARNING ||||||||||||||||||||||||||||')
with warnings.catch_warnings():
    if not verbose:
        warnings.filterwarnings("ignore", category=UserWarning)
    from nibabel.nicom import dicomreaders

if verbose:
    print(''
        '|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||')

# obtain full path for current directory: '.../rt-cloud/projects/sample'
currPath = '/gpfs/milgram/pi/turk-browne/projects/rt-cloud' #os.path.dirname(os.path.realpath(__file__))
# obtain full path for root directory: '.../rt-cloud'
rootPath = os.path.dirname(os.path.dirname(currPath))

# add the path for the root directory to your python path so that you can import
#   project modules from rt-cloud
sys.path.append(rootPath)
from rtCommon.utils import loadConfigFile, stringPartialFormat
from rtCommon.clientInterface import ClientInterface
from rtCommon.imageHandling import readRetryDicomFromDataInterface, convertDicomImgToNifti
from rtCommon.dataInterface import DataInterface #added by QL
from recognition_dataAnalysisFunctions import normalize,classifierProb


# obtain the full path for the configuration toml file
# defaultConfig = os.path.join(currPath, 'conf/sample.toml')
defaultConfig = '/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/conf/sub004.ses3.toml'



cfg = cfg_loading(defaultConfig)
# dicomData = dataInterface.getImageData(streamId, int(this_TR), timeout_file)

clientInterfaces = ClientInterface(yesToPrompts=False)
#dataInterface = clientInterfaces.dataInterface
subjInterface = clientInterfaces.subjInterface
webInterface  = clientInterfaces.webInterface

## Added by QL
allowedDirs = ['*'] #['/gpfs/milgram/pi/turk-browne/projects/rt-cloud/projects/sample/dicomDir/20190219.0219191_faceMatching.0219191_faceMatching','/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/sample', '/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/sample/dicomDir']
allowedFileTypes = ['*'] #['.txt', '.dcm']
dataInterface = DataInterface(dataRemote=False,allowedDirs=allowedDirs,allowedFileTypes=allowedFileTypes) # Create an instance of local datainterface

# Also try the placeholder for bidsInterface (an upcoming feature)
bidsInterface = clientInterfaces.bidsInterface
# res = bidsInterface.echo("test")

# doRuns(cfg, dataInterface, subjInterface, webInterface)

scanNum = int(sys.argv[1]) #12 #cfg.scanNum[0]
runNum = scanNum-1 #cfg.runNum[0]

print(f"Doing run {runNum}, scan {scanNum}")
print(f"cfg.dicomDir={cfg.dicomDir}")

allowedFileTypes = dataInterface.getAllowedFileTypes()
print(""
        "-----------------------------------------------------------------------------\n"
        "Before continuing, we need to make sure that dicoms are allowed. To verify\n"
        "this, use the 'allowedFileTypes'.\n"
        "Allowed file types: %s" %allowedFileTypes)

dicomScanNamePattern = stringPartialFormat(cfg.dicomNamePattern, 'SCAN', scanNum)
print(f"dicomScanNamePattern={dicomScanNamePattern}")

streamId = dataInterface.initScannerStream(cfg.dicomDir, 
                                                dicomScanNamePattern,
                                                cfg.minExpectedDicomSize)

tmp_dir=f"{cfg.tmp_folder}{time.time()}/" ; mkdir(tmp_dir)

tmp_dir

mask = np.load(f"{cfg.chosenMask}")

BC_clf=joblib.load(cfg.usingModel_dir +'benchchair_chairtable.joblib') # These 4 clf are the same: bedbench_benchtable.joblib bedtable_tablebench.joblib benchchair_benchtable.joblib chairtable_tablebench.joblib
BD_clf=joblib.load(cfg.usingModel_dir +'bedchair_chairbench.joblib') #

logTimes=[]
# where the morphParams are saved
# output_textFilename = f'{cfg.feedback_dir}B_probs_{scanNum}.txt'
output_matFilename = os.path.join(f'{cfg.feedback_dir}B_probs_{scanNum}.mat')

num_total_trials=12
num_total_TRs = int((num_total_trials*28+12)/2) + 8  # number of TRs to use for example 1
# morphParams = np.zeros((num_total_TRs, 1))
B_probs=[]
maskedData=0
timeout_file = 5 # small number because of demo, can increase for real-time
processedTime=[] # for


for this_TR in range(1,177):
    print(f"milgramTR_ID={this_TR}")
    dicomFilename = dicomScanNamePattern.format(TR=this_TR)
    processing_start_time=time.time()
    print(f"{cfg.dicom_dir}/{dicomFilename}")
    dicomData = dataInterface.getImageData(streamId, int(this_TR), timeout_file)
    dicomData.convert_pixel_data()
    niftiObject = dicomreaders.mosaic_to_nii(dicomData)



    niiFileName= tmp_dir+cfg.dicomNamePattern.format(SCAN=scanNum,TR=this_TR).split('.')[0]
    print(f"niiFileName={niiFileName}.nii",end='\n\n')
    nib.save(niftiObject, f"{niiFileName}.nii")  

    command=f"3dresample \
        -master {cfg.templateFunctionalVolume_converted} \
        -prefix {niiFileName}_reorient.nii \
        -input {niiFileName}.nii"
    # print(command)
    call(command,shell=True)

    command = f"3dvolreg \
            -base {cfg.templateFunctionalVolume_converted} \
            -prefix  {niiFileName}_aligned.nii \
            {niiFileName}_reorient.nii"

    # print(command)
    call(command,shell=True)

    niftiObject = nib.load(f"{niiFileName}_aligned.nii")
    nift_data = niftiObject.get_fdata()
    print(f"nift_data.shape={nift_data.shape}")
