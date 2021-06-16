# coding=UTF-8
import os
import sys
sys.path.append('/gpfs/milgram/project/turk-browne/projects/rtSynth_rt/')
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
argParser.add_argument('--config', '-c', default='sub001.ses1.toml', type=str, help='experiment file (.json or .toml)')
argParser.add_argument('--skipPre', '-s', default=False, action='store_true', help='skip preprocess or not')
argParser.add_argument('--skipGreedy', '-g', default=0, type=int, help='skip greedy or not')
argParser.add_argument('--forceGreedy', default=False, action='store_true', help='whether to force Greedy search in current session')
argParser.add_argument('--testRun', '-t', default=None, type=int, help='testRun, can be [None,1,2,3,4,5,6,7,8]')
argParser.add_argument('--scan_asTemplate', '-a', default=1, type=int, help="which scan's middle dicom as Template?")
argParser.add_argument('--preprocessOnly', default=False, action='store_true', help='whether to only do preprocess and skip everything else')
argParser.add_argument('--LeaveOutRun', '-l', default=None, type=int, help='testRun, can be [None,1,2,3,4,5,6,7,8]')

argParser.add_argument('--tmp_folder', default='_', type=str, help='tmp_folder')

args = argParser.parse_args()
from rtCommon.cfg_loading import mkdir,cfg_loading
# config="sub001.ses2.toml"
cfg = cfg_loading(args.config)

sys.path.append('/gpfs/milgram/project/turk-browne/projects/rtSynth_rt/expScripts/recognition/')
from recognition_dataAnalysisFunctions import behaviorDataLoading,normalize,append_file

def wait(waitfor, delay=1):
    while not os.path.exists(waitfor):
        time.sleep(delay)
        print('waiting for {}'.format(waitfor))

def greedyMask(cfg,N=78,LeaveOutRun=1,recordingTxt = "", tmp_folder=''): # N used to be 31, 25 
    import os
    import numpy as np
    import nibabel as nib
    import sys
    sys.path.append('/gpfs/milgram/project/turk-browne/projects/rtSynth_rt/')
    import time
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    import itertools
    # from tqdm import tqdm
    import pickle5 as pickle
    import subprocess
    from subprocess import call
    def save_obj(obj, name):
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load_obj(name):
        with open(name + '.pkl', 'rb') as f:
            return pickle.load(f)

    # What subject are you running
    '''
    Takes args (in order):
        subject (e.g. sub001)
        dataSource (e.g. realtime)
        roiloc (wang2014 or schaefer2018)
        N (the number of parcels or ROIs to start with)
    '''


    from rtCommon.cfg_loading import mkdir,cfg_loading
    # config="sub001.ses1.toml"
    # cfg = cfg_loading(config)

    subject,dataSource,roiloc,N=cfg.subjectName,"realtime","schaefer2018",N
    # subject,dataSource,roiloc,N=sys.argv[1],sys.argv[2],sys.argv[3],int(sys.argv[4])

    print("Running subject {}, with {} as a data source, {}, starting with {} ROIs".format(subject, dataSource, roiloc, N))

    # funcdata = cfg.recognition_dir + "brain_run{run}.npy"
    # metadata = cfg.recognition_dir + "behav_run{run}.csv"

    topN = load_obj(f"{cfg.recognition_expScripts_dir}top{N}ROIs")
    print(f"len(topN)={len(topN)}")
    print(f"GMschaefer_ topN loaded from neurosketch={topN}")

    def Wait(waitfor, delay=1):
        while not os.path.exists(waitfor):
            time.sleep(delay)
            print('waiting for {}'.format(waitfor))

    imcodeDict={"A": "bed", "B": "Chair", "C": "table", "D": "bench"}
    if recordingTxt=='':
        recordingTxt=f"{tmp_folder}/recording.txt"
    def getMask(topN, cfg):
        for pn, parc in enumerate(topN):
            _mask = nib.load(f"{cfg.subjects_dir}{cfg.subjectName}/ses1/recognition/mask/GMschaefer_{parc}")
            # schaefer_56.nii.gz
            aff = _mask.affine
            _mask = _mask.get_data()
            _mask = _mask.astype(int)
            # say some things about the mask.
            mask = _mask if pn == 0 else mask + _mask
            mask[mask>0] = 1
        return mask

    mask=getMask(topN, cfg)

    print('mask dimensions: {}'. format(mask.shape))
    print('number of voxels in mask: {}'.format(np.sum(mask)))


    runRecording = pd.read_csv(f"{cfg.recognition_dir}../runRecording.csv")
    actualRuns = list(runRecording['run'].iloc[list(np.where(1==1*(runRecording['type']=='recognition'))[0])]) # can be [1,2,3,4,5,6,7,8] or [1,2,4,5]
    if len(actualRuns) < 8:
        runRecording_preDay = pd.read_csv(f"{cfg.subjects_dir}{cfg.subjectName}/ses{cfg.session-1}/recognition/../runRecording.csv")
        actualRuns_preDay = list(runRecording_preDay['run'].iloc[list(np.where(1==1*(runRecording_preDay['type']=='recognition'))[0])])[-(8-len(actualRuns)):] # might be [5,6,7,8]
    else: 
        actualRuns_preDay = []

    # assert len(actualRuns_preDay)+len(actualRuns)==8 
    if len(actualRuns_preDay)+len(actualRuns)<8:
        runRecording_prepreDay = pd.read_csv(f"{cfg.subjects_dir}{cfg.subjectName}/ses{cfg.session-2}/recognition/../runRecording.csv")
        actualRuns_prepreDay = list(runRecording_prepreDay['run'].iloc[list(np.where(1==1*(runRecording_prepreDay['type']=='recognition'))[0])])[-(8-len(actualRuns)-len(actualRuns_preDay)):] # might be [5,6,7,8]
    else:
        actualRuns_prepreDay = []

    objects = ['bed', 'bench', 'chair', 'table']

    brain_data=[]
    behav_data=[]
    actualRuns.remove(actualRuns[LeaveOutRun-1])
    print(f"actualRuns={actualRuns} after removal")

    for ii,run in enumerate(actualRuns): # load behavior and brain data for current session
        t = np.load(f"{cfg.recognition_dir}brain_run{run}.npy")
        t = normalize(t)
        brain_data.append(t)

        t = pd.read_csv(f"{cfg.recognition_dir}behav_run{run}.csv")
        t=list(t['Item'])
        behav_data.append(t)
    
    if tmp_folder=='' or tmp_folder=='_':
        tmp_folder=f"tmp__folder_{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))}"
    print(f"tmp_folder={tmp_folder}")
    mkdir(f"{cfg.projectDir}{tmp_folder}")
    save_obj([brain_data,behav_data],f"{cfg.projectDir}{tmp_folder}/{subject}_{dataSource}_{roiloc}_{N}") #{len(topN)}_{i}

    def wait(tmpFile):
        while not os.path.exists(tmpFile+'_result.npy'):
            time.sleep(5)
            print(f"waiting for {tmpFile}_result.npy\n")
        time.sleep(2)
        return np.load(tmpFile+'_result.npy',allow_pickle=True)

    def numOfRunningJobs():
        # subprocess.Popen(['squeue -u kp578 | wc -l > squeue.txt'],shell=True) # sl_result = Class(_runs, bcvar)
        randomID=str(time.time())
        # print(f"squeue -u kp578 | wc -l > squeue/{randomID}.txt")
        call(f'squeue -u kp578 | wc -l > {cfg.projectDir}squeue/{randomID}.txt',shell=True)
        numberOfJobsRunning = int(open(f"{cfg.projectDir}squeue/{randomID}.txt", "r").read())
        print(f"numberOfJobsRunning={numberOfJobsRunning}")
        return numberOfJobsRunning

    def Class(brain_data,behav_data):
        # metas = bcvar[0]
        # data4d = data[0]
        print([t.shape for t in brain_data])

        accs = []
        for run in range(len(brain_data)):
            testX = brain_data[run]
            testY = behav_data[run]

            trainX=np.zeros((1,1))
            for i in range(len(brain_data)):
                if i !=run:
                    trainX=brain_data[i] if trainX.shape==(1,1) else np.concatenate((trainX,brain_data[i]),axis=0)

            trainY = []
            for i in range(len(brain_data)):
                if i != run:
                    trainY.extend(behav_data[i])
            clf = LogisticRegression(penalty='l2',C=1, solver='lbfgs', max_iter=1000, 
                                    multi_class='multinomial').fit(trainX, trainY)
                    
            # Monitor progress by printing accuracy (only useful if you're running a test set)
            acc = clf.score(testX, testY)
            accs.append(acc)
        
        return np.mean(accs)

    if not os.path.exists(f"{cfg.projectDir}{tmp_folder}/{subject}_{N}_{roiloc}_{dataSource}_{len(topN)}.pkl"):
        brain_data = [t[:,mask==1] for t in brain_data]
        # _runs = [runs[:,mask==1]]
        print("Runs shape", [t.shape for t in brain_data])
        slstart = time.time()
        sl_result = Class(brain_data, behav_data)
        print(f"passed {time.time()-slstart}s for training")
        save_obj({"subject":subject,
        "startFromN":N,
        "currNumberOfROI":len(topN),
        "bestAcc":sl_result, # this is the sl_result for the topN, not the bestAcc, bestAcc is for the purpose of keeping consistent with others
        "bestROIs":topN},# this is the topN, not the bestROIs, bestROIs is for the purpose of keeping consistent with others
        f"{cfg.projectDir}{tmp_folder}/{subject}_{N}_{roiloc}_{dataSource}_{len(topN)}"
        )

    # if os.path.exists(f"{cfg.projectDir}{tmp_folder}/{subject}_{N}_{roiloc}_{dataSource}_{1}.pkl"):
    #     print(f"{cfg.projectDir}{tmp_folder}/{subject}_{N}_{roiloc}_{dataSource}_1.pkl exists")
    #     return recordingTxt
    #     raise Exception('runned or running')

    # N-1
    def next(topN):
        print(f"len(topN)={len(topN)}")
        print(f"topN={topN}")

        if len(topN)==1:
            return None
        else:
            allpairs = itertools.combinations(topN,len(topN)-1)
            topNs=[]
            sl_results=[]
            tmpFiles=[]
            while os.path.exists(f"{cfg.projectDir}{tmp_folder}/holdon.npy"):
                time.sleep(10)
                print(f"sleep for 10s ; waiting for ./{tmp_folder}/holdon.npy to be deleted")
            np.save(f"{cfg.projectDir}{tmp_folder}/holdon",1)

            # 对于每一个round，提交一个job array，然后等待这个job array完成之后再进行下一轮
            # 具体的方法是首先保存需要的input，也就是这一轮需要用到的tmpFile，然后再将tmpFile除了之外的字符串输入
            skip_flag=0
            for i,_topN in enumerate(allpairs):
                tmpFile=f"{cfg.projectDir}{tmp_folder}/{subject}_{N}_{roiloc}_{dataSource}_{len(topN)}_{i}"
                print(f"tmpFile={tmpFile}")
                topNs.append(_topN)
                tmpFiles.append(tmpFile)

                if not os.path.exists(tmpFile+'_result.npy'):
                    # prepare brain data(runs) mask and behavior data(bcvar) 
                    save_obj([_topN,subject,dataSource,roiloc,N], tmpFile)
                else:
                    print(tmpFile+'_result.npy exists!')
                    skip_flag+=1

            if skip_flag!=(i+1): # 如果有一个不存在，就需要跑一跑
                command=f'sbatch --array=1-{i+1} {cfg.recognition_expScripts_dir}class_LOO.sh ./{tmp_folder}/{subject}_{N}_{roiloc}_{dataSource}_{len(topN)}_'
                print(command)
                proc = subprocess.Popen([command], shell=True) # sl_result = Class(_runs, bcvar) 
            else:
                command=f'sbatch --array=1-{i+1} {cfg.recognition_expScripts_dir}class_LOO.sh ./{tmp_folder}/{subject}_{N}_{roiloc}_{dataSource}_{len(topN)}_'
                print(f"skip {command}")
            try:
                os.remove(f"{cfg.projectDir}{tmp_folder}/holdon.npy")
            except:
                pass
            # wait for everything to be finished and make a summary to find the best performed megaROI
            sl_results=[]
            for tmpFile in tmpFiles:
                sl_result=wait(tmpFile)
                sl_results.append(sl_result)
            print(f"sl_results={sl_results}")
            print(f"max(sl_results)=={max(sl_results)}")
            maxID=np.where(sl_results==max(sl_results))[0][0]
            save_obj({"subject":subject,
            "startFromN":N,
            "currNumberOfROI":len(topN)-1,
            "bestAcc":max(sl_results),
            "bestROIs":topNs[maxID]},
            f"{cfg.projectDir}{tmp_folder}/{subject}_{N}_{roiloc}_{dataSource}_{len(topN)-1}"
            )
            print(f"bestAcc={max(sl_results)} For {len(topN)-1} = {cfg.projectDir}{tmp_folder}/{subject}_{N}_{roiloc}_{dataSource}_{len(topN)-1}")
            tmpFiles=next(topNs[maxID])
            return 0
    tmpFiles=next(topN)



    # when every mask has run, find the best mask and save as the chosenMask
    roiloc="schaefer2018"
    dataSource="realtime"
    subjects=[cfg.subjectName]
    N=N
    GreedyBestAcc=np.zeros((len(subjects),N+1))
    GreedyBestAcc[GreedyBestAcc==0]=None
    for ii,subject in enumerate(subjects):
        for len_topN_1 in range(N-1,0,-1):
            try:
                # print(f"./{tmp_folder}/{subject}_{N}_{roiloc}_{dataSource}_{len_topN_1}")
                di = load_obj(f"{cfg.projectDir}{tmp_folder}/{subject}_{N}_{roiloc}_{dataSource}_{len_topN_1}")
                GreedyBestAcc[ii,len_topN_1-1] = di['bestAcc']
            except:
                pass
    GreedyBestAcc=GreedyBestAcc.T

    # import matplotlib.pyplot as plt
    # plt.imshow(GreedyBestAcc)
    # _=plt.figure()
    # for i in range(GreedyBestAcc.shape[0]):
    #     plt.scatter([i]*GreedyBestAcc.shape[1],GreedyBestAcc[i],c='g',s=2)
    # plt.plot(np.arange(GreedyBestAcc.shape[0]),np.nanmean(GreedyBestAcc,axis=1))

    performance_mean = np.nanmean(GreedyBestAcc,axis=1)
    bestID=np.where(performance_mean==max(performance_mean))[0][0]
    di = load_obj(f"./{tmp_folder}/{subject}_{N}_{roiloc}_{dataSource}_{bestID+1}")
    print(f"bestID={bestID}; best Acc = {di['bestAcc']}")
    print(f"bestROIs={di['bestROIs']}")
    
    append_file(recordingTxt,f"bestID={bestID}; best Acc = {di['bestAcc']}")
    append_file(recordingTxt,f"bestROIs={di['bestROIs']}")

    mask = getMask(di['bestROIs'],cfg)
    np.save(f"{cfg.recognition_dir}chosenMask_leave_{LeaveOutRun}_out.npy",mask)
    
    
    
    return recordingTxt


def minimalClass(cfg,LeaveOutRun=1,recordingTxt=None):
    '''
    purpose: 
        train offline models

    steps:
        load preprocessed and aligned behavior and brain data 
        select data with the wanted pattern like AB AC AD BC BD CD 
        train correspondng classifier and save the classifier performance and the classifiers themselves.
    '''

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import sklearn
    import joblib
    import nibabel as nib
    import itertools
    from sklearn.linear_model import LogisticRegression

    def other(target):
        other_objs = [i for i in ['bed', 'bench', 'chair', 'table'] if i not in target]
        return other_objs

    def red_vox(n_vox, prop=0.1):
        return int(np.ceil(n_vox * prop))

    if 'milgram' in os.getcwd():
        main_dir='/gpfs/milgram/project/turk-browne/projects/rtSynth_rt/'
    else:
        main_dir='/Volumes/GoogleDrive/My Drive/Turk_Browne_Lab/rtcloud_kp/'

    working_dir=main_dir
    os.chdir(working_dir)

    '''
    if you read runRecording for current session and found that there are only 4 runs in the current session, 
    you read the runRecording for previous session and fetch the last 4 recognition runs from previous session
    '''
    runRecording = pd.read_csv(f"{cfg.recognition_dir}../runRecording.csv")
    actualRuns = list(runRecording['run'].iloc[list(np.where(1==1*(runRecording['type']=='recognition'))[0])]) # can be [1,2,3,4,5,6,7,8] or [1,2,4,5]
    # if len(actualRuns) < 8:
    #     runRecording_preDay = pd.read_csv(f"{cfg.subjects_dir}{cfg.subjectName}/ses{cfg.session-1}/recognition/../runRecording.csv")
    #     actualRuns_preDay = list(runRecording_preDay['run'].iloc[list(np.where(1==1*(runRecording_preDay['type']=='recognition'))[0])])[-(8-len(actualRuns)):] # might be [5,6,7,8]
    # else: 
    #     actualRuns_preDay = []

    # # assert len(actualRuns_preDay)+len(actualRuns)==8 
    # if len(actualRuns_preDay)+len(actualRuns)<8:
    #     runRecording_prepreDay = pd.read_csv(f"{cfg.subjects_dir}{cfg.subjectName}/ses{cfg.session-2}/recognition/../runRecording.csv")
    #     actualRuns_prepreDay = list(runRecording_prepreDay['run'].iloc[list(np.where(1==1*(runRecording_prepreDay['type']=='recognition'))[0])])[-(8-len(actualRuns)-len(actualRuns_preDay)):] # might be [5,6,7,8]
    # else:
    #     actualRuns_prepreDay = []

    objects = ['bed', 'bench', 'chair', 'table']

    new_run_indexs=[]
    new_run_index=1 #使用新的run 的index，以便于后面的testRun selection的时候不会重复。正常的话 new_run_index 应该是1，2，3，4，5，6，7，8
    for ii,run in enumerate(actualRuns): # load behavior and brain data for current session
        t = np.load(f"{cfg.recognition_dir}brain_run{run}.npy")
        mask = np.load(f"{cfg.recognition_dir}chosenMask_leave_{LeaveOutRun}_out.npy")
        print(f"loading {cfg.recognition_dir}chosenMask_leave_{LeaveOutRun}_out.npy")
        
        t = t[:,mask==1]
        t = normalize(t)
        brain_data=t if ii==0 else np.concatenate((brain_data,t), axis=0)

        t = pd.read_csv(f"{cfg.recognition_dir}behav_run{run}.csv")
        t['run_num'] = new_run_index
        new_run_indexs.append(new_run_index)
        new_run_index+=1
        behav_data=t if ii==0 else pd.concat([behav_data,t])

    FEAT=brain_data.reshape(brain_data.shape[0],-1)

    FEAT=brain_data
    print(f"FEAT.shape={FEAT.shape}")
    assert len(FEAT.shape)==2
    # FEAT_mean=np.mean(FEAT,axis=1)
    # FEAT=(FEAT.T-FEAT_mean).T
    # FEAT_mean=np.mean(FEAT,axis=0)
    # FEAT=FEAT-FEAT_mean
    # FEAT = normalize(FEAT)

    META=behav_data

    # convert item colume to label colume
    imcodeDict={
    'A': 'bed',
    'B': 'chair',
    'C': 'table',
    'D': 'bench'}
    label=[]
    for curr_trial in range(META.shape[0]):
        label.append(imcodeDict[META['Item'].iloc[curr_trial]])
    META['label']=label # merge the label column with the data dataframe

    # Which run to use as test data (leave as None to not have test data)
    # testRun = 0 # when testing: testRun = 2 ; META['run_num'].iloc[:5]=2
    def train4wayClf(META, FEAT):
        runList = np.unique(list(META['run_num']))
        print(f"runList={runList}")
        accList={}
        for testRun in runList:
            trainIX = META['run_num']!=int(testRun)
            testIX = META['run_num']==int(testRun)

            # pull training and test data
            trainX = FEAT[trainIX]
            testX = FEAT[testIX]
            trainY = META.iloc[np.asarray(trainIX)].label
            testY = META.iloc[np.asarray(testIX)].label

            # Train your classifier
            clf = LogisticRegression(penalty='l2',C=1, solver='lbfgs', max_iter=1000, 
                                        multi_class='multinomial').fit(trainX, trainY)
            
            # model_folder = cfg.trainingModel_dir
            # Save it for later use
            # joblib.dump(clf, model_folder +'/{}.joblib'.format(naming))
            
            # Monitor progress by printing accuracy (only useful if you're running a test set)
            acc = clf.score(testX, testY)
            print("acc=", acc)
            accList[testRun] = acc
        print(f"new trained full rotation 4 way accuracy mean={np.mean(list(accList.values()))}")
        if recordingTxt: #if tmp_folder is not None but some string, save the sentence.
            append_file(f"{recordingTxt}",f"new trained full rotation 4 way accuracy mean={np.mean(list(accList.values()))}")
        
        return accList
    accList = train4wayClf(META, FEAT)
    
    # 获得full rotation的2way clf的accuracy 平均值 中文
    accs_rotation=[]
    print(f"new_run_indexs={new_run_indexs}")

    # for testRun in new_run_indexs:
    testRun=LeaveOutRun
    allpairs = itertools.combinations(objects,2)
    accs={}
    # Iterate over all the possible target pairs of objects
    for pair in allpairs:
        # Find the control (remaining) objects for this pair
        altpair = other(pair)
        
        # pull sorted indices for each of the critical objects, in order of importance (low to high)
        # inds = get_inds(FEAT, META, pair, testRun=testRun)
        
        # Find the number of voxels that will be left given your inclusion parameter above
        # nvox = red_vox(FEAT.shape[1], include)
        
        for obj in pair:
            # foil = [i for i in pair if i != obj][0]
            for altobj in altpair:
                # establish a naming convention where it is $TARGET_$CLASSIFICATION
                # Target is the NF pair (e.g. bed/bench)
                # Classificationis is btw one of the targets, and a control (e.g. bed/chair, or bed/table, NOT bed/bench)
                naming = '{}{}_{}{}'.format(pair[0], pair[1], obj, altobj)

                if testRun:
                    trainIX = ((META['label']==obj) | (META['label']==altobj)) & (META['run_num']!=int(testRun))
                    testIX = ((META['label']==obj) | (META['label']==altobj)) & (META['run_num']==int(testRun))
                else:
                    trainIX = ((META['label']==obj) | (META['label']==altobj))
                    testIX = ((META['label']==obj) | (META['label']==altobj))

                # pull training and test data
                trainX = FEAT[trainIX]
                testX = FEAT[testIX]
                trainY = META.iloc[np.asarray(trainIX)].label
                testY = META.iloc[np.asarray(testIX)].label

                assert len(np.unique(trainY))==2

                # Train your classifier
                clf = LogisticRegression(penalty='l2',C=1, solver='lbfgs', max_iter=1000, 
                                            multi_class='multinomial').fit(trainX, trainY)
                
                model_folder = cfg.trainingModel_dir
                # Save it for later use
                # joblib.dump(clf, model_folder +'/{}.joblib'.format(naming))
                
                # Monitor progress by printing accuracy (only useful if you're running a test set)
                acc = clf.score(testX, testY)
                print(naming, acc)
                accs[naming]=acc

    print(f"accs={accs}")
    print(f"LeaveOutRun = {LeaveOutRun} : average 2 way clf accuracy={np.mean(list(accs.values()))}")
    # accs_rotation.append(np.mean(list(accs.values())))
    # print(f"mean of 2 way clf acc full rotation = {np.mean(accs_rotation)}")
    if recordingTxt: #if tmp_folder is not None but some string, save the sentence.
        append_file(f"{recordingTxt}",f"accs={accs}")
        append_file(f"{recordingTxt}",f"LeaveOutRun = {LeaveOutRun} : average 2 way clf accuracy={np.mean(list(accs.values()))}")
    
    return accs

# recordingTxt=f"{cfg.subjects_dir}{cfg.subjectName}/ses{cfg.session}/recognition/recording.txt" # None
forceGreedy="forceGreedy"
recordingTxt=''
if args.tmp_folder=="_":
    tmp_folder=''
else:
    tmp_folder=args.tmp_folder

# for LeaveOutRun in range(1,9):
LeaveOutRun=args.LeaveOutRun
print(f"LeaveOutRun={LeaveOutRun}")

recordingTxt=greedyMask(cfg, LeaveOutRun=int(LeaveOutRun),recordingTxt=recordingTxt,tmp_folder=tmp_folder)
accs = minimalClass(cfg,LeaveOutRun=LeaveOutRun,recordingTxt=recordingTxt)
print("\n\n")
print(f"minimalClass accs={accs}")
# save_obj(accs,f"{cfg.recognition_dir}minimalClass_accs")
save_obj(accs,f"{cfg.recognition_dir}/Leave_{LeaveOutRun}_out_Greedy_and_trainTest")

