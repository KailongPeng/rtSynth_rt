#-*- coding: utf-8 -*-
import os,time,shutil
import numpy as np
from glob import glob
def find_ABCD_T1w_MPR_vNav(sub): 
    """这个函数的功能是找到第二个ABCD_T1w_MPR_vNav  usable的前面的数字，保存在ABCD_T1w_MPR_vNav.txt里面"""

    raw_dir="/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/expScripts/recognition/recognitionDataAnalysis/raw/"
    os.chdir(raw_dir)
    f = open(raw_dir+sub+"_run_name.txt","r")
    d = f.read()
    T1_ID = d.split("ABCD_T1w_MPR_vNav")[3]
    T1_ID = int(T1_ID.split("\n")[1])

    T2_ID = d.split("ABCD_T2w_SPC_vNav")[3]
    T2_ID = int(T2_ID.split("\n")[1])

    f = open(raw_dir+sub+"_ABCD_T1w_MPR_vNav.txt","w")
    f.write("T1_ID={} ; T2_ID={}".format(T1_ID,T2_ID))
    f.close()

def wait(waitfor, delay=1):
    while not os.path.exists(waitfor):
        time.sleep(delay)
        print('waiting for {}'.format(waitfor))
        
def find_T1_in_niiFolder(T1_ID,T2_ID,sub):
    subjectName=sub.split("_")[1] # sub 可能是rtSynth_sub001_ses5 或者 rtSynth_sub001; subjectName 应该是sub001之类的
    if len(sub.split("_"))==3:
        ses=int(sub.split("_")[2].split("ses")[1]) #ses应该是数字
    else:
        ses=1

    nii_path="/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/expScripts/recognition/recognitionDataAnalysis/raw/{}/nifti/".format(sub)
    print("nii_path={}".format(nii_path))
    files = glob("{}*.nii".format(nii_path))
    for curr_file in files:
        runID=curr_file.split("/")[-1].split("_")[-1].split(".nii")[0]
        try:
            if int(runID)==T1_ID:
                print(curr_file)
                T1=curr_file
            elif int(runID)==T2_ID:
                print(curr_file)
                T2=curr_file
        except:
            pass
    AnatFolder = "/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/subjects/{}/ses{}/anat/".format(subjectName,ses)
    print("T1={}".format(T1))
    print("T2={}".format(T2))
    print("AnatFolder={}".format(AnatFolder))
    shutil.copyfile(T1,"{}T1.nii".format(AnatFolder))
    shutil.copyfile(T2,"{}T2.nii".format(AnatFolder))

def _split(sub):
    subjectName=sub.split("_")[1] # sub 可能是rtSynth_sub001_ses5 或者 rtSynth_sub001; subjectName 应该是sub001之类的
    if len(sub.split("_"))==3:
        ses=int(sub.split("_")[2].split("ses")[1]) #ses应该是数字
    else:
        ses=1

    code_dir="/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/expScripts/recognition/recognitionDataAnalysis/"
    f = open("{}{}_subjectName.txt".format(code_dir,sub),"w")
    f.write("subjectName={} ; ses={}".format(subjectName,ses))
    f.close()
