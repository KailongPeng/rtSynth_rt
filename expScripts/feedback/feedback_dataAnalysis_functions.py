def prob_analysis():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    sub='sub003'
    ls=[]
    probs=[]
    for curr_ses in range(2,5):
        # for currRun in range(1,11):
        currRun=1
        while True:
            try:
                history=pd.read_csv(f"/Users/kailong/Desktop/rtEnv/rtSynth_rt/subjects/{sub}/ses{curr_ses}/feedback/{sub}_{currRun}_history.csv")
                l = list(history[history['states']=="feedback"]['B_prob'])
                ls.append(np.mean(l))
                # print(np.mean(l))
                if len(probs)==0:
                    probs = np.expand_dims(l,0)
                else:
                    probs = np.concatenate([probs,np.expand_dims(l,0)],axis=0) 
                currRun+=1
            except:
                break
    _=plt.figure()
    plt.plot(ls)
    plt.xlabel("run ID")
    plt.ylabel("mean prob of only feedback TRs in that run")
    plt.title("mean prob of only feedback TRs in that run v.s. run ID")
    print(f"mean prob of all feedback TRs={np.mean(ls)}")





    import seaborn as sns
    # sns.set_theme(style="whitegrid")
    # tips = sns.load_dataset("tips")
    # ax = sns.boxplot(x=probs)
    # np.expand_dims(l,0).shape
    # probs.shape
    # sns.boxplot(probs)
    _=plt.figure(figsize=(20,20))
    _=plt.boxplot(probs.T)
    for currRun in range(len(probs)):
        plt.scatter([currRun+1+0.1]*60,probs[currRun],s=1)
    _=plt.xlabel("run ID")
    _=plt.ylabel("prob")
    _=plt.plot(np.arange(1,len(probs)+1),ls)





    import pandas as pd
    import numpy as np
    from sklearn import datasets, linear_model
    from sklearn.linear_model import LinearRegression
    import statsmodels.api as sm
    from scipy import stats

    y = probs.reshape(np.prod(probs.shape))
    X=[]
    for currRun in range(1,len(probs)+1):
        X+=[currRun]*60

    X2 = sm.add_constant(X)
    est = sm.OLS(y, X2)
    est2 = est.fit()
    print(est2.summary())
    print("Ordinary least squares")



    def gaussian(x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    def gaussian_fit(y=[1,2,3,4],x=1):
        y_values = np.linspace(min(y), max(y), 120)    
        mu=np.mean(y)
        sig=np.std(y)
        #plt.text(x+0.04, mu, 'mu={0:.2f}\nsig={1:.2f}'.format(mu,sig), fontsize=12)
        plt.plot(x+0.04+0.5*gaussian(y_values, mu, sig),y_values)
        

    _=plt.figure(figsize=(10,10))
    for currRun in range(len(probs)):
        plt.scatter([currRun]*60,probs[currRun],s=1)
        gaussian_fit(y=probs[currRun],x=currRun)
    b=0.0061
    const=0.3880

    plt.plot(np.arange(len(probs)),np.arange(len(probs))*b+const)
    plt.xlabel("run ID")
    plt.ylabel("probability")
    plt.title("Gaussian fitted probability distribution")

# Trained axis v.s. Untrained axis
# Drift happening 
def plotForceGreedyAccCurve():
    way4 = [0.7401374113475178,0.6668513593380615,0.653125,0.5723168169904409,0.7005208333333334]
    way2 = [0.8607714371980677,0.8386754776021079,0.8017882630654369,0.7814661561264822,0.8541666666666666]
    fig,axs=plt.subplots(1,2,figsize=(14,7))
    axs[0].plot(np.arange(1,len(way4)+1),new_trained_full_rotation_4_way_accuracy_mean,label="4_way new mask",color="orange")
    axs[0].plot(np.arange(1,len(way4)+1),mean_of_2_way_clf_acc_full_rotation,label="2_way new mask",color="red")
    axs[0].plot(np.arange(1,len(way4)+1),[0.25]*len(way4),'--',color='orange')
    axs[0].plot(np.arange(1,len(way4)+1),[0.5]*len(way4),'--',color='red')
    axs[0].legend()
    axs[0].set_ylabel("acc")
    axs[0].set_xlabel("session ID")
    axs[0].set_ylim([0.24,0.9])
    axs[0].set_title("using new mask")


    way4=[0.7401374113475178,0.5740543735224587,0.5038194444444444,0.4748424491211841,0.546875]
    way2=[0.8607714371980677,0.8114603919631094,0.7822690217391304,0.7463665184453228,0.8159722222222222]
    axs[1].plot(np.arange(1,len(way4)+1),way4,label="4_way old mask",color="orange")
    axs[1].plot(np.arange(1,len(way4)+1),way2,label="2_way old mask",color="red")
    axs[1].plot(np.arange(1,len(way4)+1),[0.25]*len(way4),'--',color='orange')
    axs[1].plot(np.arange(1,len(way4)+1),[0.5]*len(way4),'--',color='red')
    axs[1].legend()
    axs[1].set_ylabel("acc")
    axs[1].set_xlabel("session ID")
    axs[1].set_ylim([0.24,0.9])
    axs[1].set_title("using ses1 mask")

def megaROIOverlapping():
    import os
    import re
    from glob import glob



    os.chdir("/gpfs/milgram/pi/turk-browne/projects/rt-cloud/projects/rtSynth_rt/")
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

    def getBestROIs(logID='17800181'):

        log = glob(f'./logs/*{logID}*')[0]
        f = open(log, "r")
        txt=f.read()
        bestROIs = txt.split("bestROIs=(")[1].split(")\n/gpfs/milgram/")[0]
        bestROIs = bestROIs.split("', '")
        bestROIs = [re.findall(r'\d+', i)[0] for i in bestROIs]

        return bestROIs
    def get4wayacc(logID='17800181'):

        log = glob(f'./logs/*{logID}*')[0]
        f = open(log, "r")
        txt=f.read()
        acc = float(txt.split("new trained full rotation 4 way accuracy mean=")[1].split("\nnew_run_indexs")[0])
        return acc
    def get2wayacc(logID='17800181'):

        log = glob(f'./logs/*{logID}*')[0]
        f = open(log, "r")
        txt=f.read()
        acc = float(txt.split("mean of 2 way clf acc full rotation =")[1].split("\nbedbench_bedchair")[0])
        return acc
    def getAB_acc(logID='17800181'):

        log = glob(f'./logs/*{logID}*')[0]
        f = open(log, "r")
        txt=f.read()
        acc = float(txt.split("mean of 2 way clf acc full rotation =")[1].split("\nbedbench_bedchair")[0])
        return acc

    logID={1:'00000181',2:'17800181',3:'17800357',4:'17806127',5:'17800522'}
    BestROIs=[]
    FourWayAcc=[]
    TwoWayAcc=[]
    AB_acc=[]
    for currSess in range(1,6):
        BestROIs.append(getBestROIs(logID=logID[currSess]))
        FourWayAcc.append(get4wayacc(logID=logID[currSess]))
        TwoWayAcc.append(get2wayacc(logID=logID[currSess]))
        AB_acc.append(getAB_acc(logID=logID[currSess]))
    BestROIs

    _bestROIs=[]
    for i in BestROIs:
        _bestROIs.append([int(a) for a in i])

    _=plt.figure(figsize=(10,10))
    for currSession in range(1,len(_bestROIs)+1):
        plt.scatter(_bestROIs[currSession-1],[currSession]*len(_bestROIs[currSession-1]),s=10,label=f"session{currSession}")
    plt.title("compare the best ROI selected from each session")    
    plt.ylabel("session")
    plt.xlabel("ROI ID")





    def inRatio(a1,a2):
        count=0
        for i in a2:
            if i in a1:
                count+=1
        return count/len(a1)
    plt.figure(figsize=(16, 8)) 
    for i in range(1,6):
        ratios=[]
        for currSession in range(1,6):
            t = inRatio(_bestROIs[i-1],_bestROIs[currSession-1])
            ratios.append(t)
        plt.subplot(1,6,i)
        plt.plot(np.arange(1,6),ratios)
        plt.xlabel("session")
        plt.ylim([0,1])
        plt.title(f"ROI in ses x out of ses{i}")



    array=np.zeros((5,300))
    for currROI in range(1,301):
        for currSes in range(1,6):
            if currROI in _bestROIs[currSes-1]:
                array[currSes-1, currROI-1] = 1
    # plt.imshow(array)
    plt.figure(figsize=(10,10))
    plt.plot(np.arange(1,301),np.sum(array,axis=0))
    plt.xlabel("ROI ID")
    plt.ylabel("count of existence")



    arrayMean=np.sum(array,axis=0)
    print(f"ID of survived ROI in all sessions={np.where(arrayMean==5)[0]+1},possible range 1-300")

    arrayMean=np.sum(array,axis=0)
    arrayMean[arrayMean==0]=None
    plt.hist(arrayMean)
    #plt.xlim([0.9,6.5])




    # cd /gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/subjects/sub003/ses1/recognition/mask
    # load_fsl
    # fslview_deprecated GMschaefer_8.nii.gz GMschaefer_159.nii.gz GMschaefer_160.nii.gz GMschaefer_163.nii.gz
    # 8 159 160 163


    def GreedySum(bestROIs=None,sub=None):
        import nibabel as nib
        import pandas as pd
        workingDir=f"/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/subjects/{sub}/ses1/recognition/mask/"
        for pn, parc in enumerate(bestROIs):
            _mask = nib.load(workingDir+f"GMschaefer_{parc}")
            aff = _mask.affine
            _mask = _mask.get_data()
            _mask = _mask.astype(int)
            mask = _mask if pn == 0 else mask + _mask
            
        savemask = nib.Nifti1Image(mask, affine=aff)
        nib.save(savemask, f"{workingDir}GreedySum.nii.gz")
        
    for i in _bestROIs:
        for j in i:
            print(f"{j}.nii.gz', '",end='')
            
    bestROIs_allSes = ('8.nii.gz', '159.nii.gz', '235.nii.gz', '163.nii.gz', '271.nii.gz', '164.nii.gz', '258.nii.gz', '80.nii.gz', '218.nii.gz', '67.nii.gz', '211.nii.gz', '2.nii.gz', '220.nii.gz', '62.nii.gz', '160.nii.gz', '22.nii.gz', '79.nii.gz', '8.nii.gz', '159.nii.gz', '114.nii.gz', '235.nii.gz', '163.nii.gz', '164.nii.gz', '151.nii.gz', '80.nii.gz', '112.nii.gz', '126.nii.gz', '67.nii.gz', '209.nii.gz', '211.nii.gz', '205.nii.gz', '2.nii.gz', '39.nii.gz', '160.nii.gz', '244.nii.gz', '8.nii.gz', '159.nii.gz', '195.nii.gz', '163.nii.gz', '164.nii.gz', '151.nii.gz', '80.nii.gz', '58.nii.gz', '67.nii.gz', '209.nii.gz', '211.nii.gz', '150.nii.gz', '160.nii.gz', '246.nii.gz', '8.nii.gz', '223.nii.gz', '159.nii.gz', '163.nii.gz', '271.nii.gz', '80.nii.gz', '126.nii.gz', '67.nii.gz', '146.nii.gz', '2.nii.gz', '160.nii.gz', '246.nii.gz', '22.nii.gz', '244.nii.gz', '30.nii.gz', '8.nii.gz', '86.nii.gz', '159.nii.gz', '195.nii.gz', '163.nii.gz', '56.nii.gz', '77.nii.gz', '76.nii.gz', '263.nii.gz', '62.nii.gz', '281.nii.gz', '160.nii.gz', '30.nii.gz', '79.nii.gz')
    GreedySum(bestROIs=bestROIs_allSes,sub='sub003')
    
    # cd /gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/subjects/sub003/ses1/recognition/mask
    # fslview_deprecated GreedySum.nii.gz ../WANGinFUNC.nii.gz templateFunctionalVolume_bet.nii.gz ../templateFunctionalVolume_bet.nii.gz

def log_analysis():
    
   # cd /gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/subjects/sub003/ses1/recognition/mask
   # load_fsl
   # fslview_deprecated GMschaefer_8.nii.gz GMschaefer_159.nii.gz GMschaefer_160.nii.gz GMschaefer_163.nii.gz
   8 159 160 163

All analysis code: /Users/kailong/Desktop/rtEnv/rtSynth_rt/expScripts/feedback/feedback_dataAnalysis.py

def prob_analysis():
   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   sub='sub003'
   ls=[]
   probs=[]
   for curr_ses in range(2,5):
       # for currRun in range(1,11):
       currRun=1
       while True:
           try:
               history=pd.read_csv(f"/Users/kailong/Desktop/rtEnv/rtSynth_rt/subjects/{sub}/ses{curr_ses}/feedback/{sub}_{currRun}_history.csv")
               l = list(history[history['states']=="feedback"]['B_prob'])
               ls.append(np.mean(l))
               # print(np.mean(l))
               if len(probs)==0:
                   probs = np.expand_dims(l,0)
               else:
                   probs = np.concatenate([probs,np.expand_dims(l,0)],axis=0)
               currRun+=1
           except:
               break
   _=plt.figure()
   plt.plot(ls)
   plt.xlabel("run ID")
   plt.ylabel("mean prob of only feedback TRs in that run")
   plt.title("mean prob of only feedback TRs in that run v.s. run ID")
   print(f"mean prob of all feedback TRs={np.mean(ls)}")





   import seaborn as sns
   # sns.set_theme(style="whitegrid")
   # tips = sns.load_dataset("tips")
   # ax = sns.boxplot(x=probs)
   # np.expand_dims(l,0).shape
   # probs.shape
   # sns.boxplot(probs)
   _=plt.figure(figsize=(20,20))
   _=plt.boxplot(probs.T)
   for currRun in range(len(probs)):
       plt.scatter([currRun+1+0.1]*60,probs[currRun],s=1)
   _=plt.xlabel("run ID")
   _=plt.ylabel("prob")
   _=plt.plot(np.arange(1,len(probs)+1),ls)





   import pandas as pd
   import numpy as np
   from sklearn import datasets, linear_model
   from sklearn.linear_model import LinearRegression
   import statsmodels.api as sm
   from scipy import stats

   y = probs.reshape(np.prod(probs.shape))
   X=[]
   for currRun in range(1,len(probs)+1):
       X+=[currRun]*60

   X2 = sm.add_constant(X)
   est = sm.OLS(y, X2)
   est2 = est.fit()
   print(est2.summary())
   print("Ordinary least squares")



   def gaussian(x, mu, sig):
       return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

   def gaussian_fit(y=[1,2,3,4],x=1):
       y_values = np.linspace(min(y), max(y), 120)   
       mu=np.mean(y)
       sig=np.std(y)
       #plt.text(x+0.04, mu, 'mu={0:.2f}\nsig={1:.2f}'.format(mu,sig), fontsize=12)
       plt.plot(x+0.04+0.5*gaussian(y_values, mu, sig),y_values)
      

   _=plt.figure(figsize=(10,10))
   for currRun in range(len(probs)):
       plt.scatter([currRun]*60,probs[currRun],s=1)
       gaussian_fit(y=probs[currRun],x=currRun)
   b=0.0061
   const=0.3880

   plt.plot(np.arange(len(probs)),np.arange(len(probs))*b+const)
   plt.xlabel("run ID")
   plt.ylabel("probability")
   plt.title("Gaussian fitted probability distribution")

# Trained axis v.s. Untrained axis
# Drift happening
def plotForceGreedyAccCurve():
   way4 = [0.7401374113475178,0.6668513593380615,0.653125,0.5723168169904409,0.7005208333333334]
   way2 = [0.8607714371980677,0.8386754776021079,0.8017882630654369,0.7814661561264822,0.8541666666666666]
   fig,axs=plt.subplots(1,2,figsize=(14,7))
   axs[0].plot(np.arange(1,len(way4)+1),new_trained_full_rotation_4_way_accuracy_mean,label="4_way new mask",color="orange")
   axs[0].plot(np.arange(1,len(way4)+1),mean_of_2_way_clf_acc_full_rotation,label="2_way new mask",color="red")
   axs[0].plot(np.arange(1,len(way4)+1),[0.25]*len(way4),'--',color='orange')
   axs[0].plot(np.arange(1,len(way4)+1),[0.5]*len(way4),'--',color='red')
   axs[0].legend()
   axs[0].set_ylabel("acc")
   axs[0].set_xlabel("session ID")
   axs[0].set_ylim([0.24,0.9])
   axs[0].set_title("using new mask")


   way4=[0.7401374113475178,0.5740543735224587,0.5038194444444444,0.4748424491211841,0.546875]
   way2=[0.8607714371980677,0.8114603919631094,0.7822690217391304,0.7463665184453228,0.8159722222222222]
   axs[1].plot(np.arange(1,len(way4)+1),way4,label="4_way old mask",color="orange")
   axs[1].plot(np.arange(1,len(way4)+1),way2,label="2_way old mask",color="red")
   axs[1].plot(np.arange(1,len(way4)+1),[0.25]*len(way4),'--',color='orange')
   axs[1].plot(np.arange(1,len(way4)+1),[0.5]*len(way4),'--',color='red')
   axs[1].legend()
   axs[1].set_ylabel("acc")
   axs[1].set_xlabel("session ID")
   axs[1].set_ylim([0.24,0.9])
   axs[1].set_title("using ses1 mask")

def megaROIOverlapping():
   import os
   import re
   from glob import glob



   os.chdir("/gpfs/milgram/pi/turk-browne/projects/rt-cloud/projects/rtSynth_rt/")
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

   def getBestROIs(logID='17800181'):

       log = glob(f'./logs/*{logID}*')[0]
       f = open(log, "r")
       txt=f.read()
       bestROIs = txt.split("bestROIs=(")[1].split(")\n/gpfs/milgram/")[0]
       bestROIs = bestROIs.split("', '")
       bestROIs = [re.findall(r'\d+', i)[0] for i in bestROIs]

       return bestROIs
   def get4wayacc(logID='17800181'):

       log = glob(f'./logs/*{logID}*')[0]
       f = open(log, "r")
       txt=f.read()
       acc = float(txt.split("new trained full rotation 4 way accuracy mean=")[1].split("\nnew_run_indexs")[0])
       return acc
   def get2wayacc(logID='17800181'):

       log = glob(f'./logs/*{logID}*')[0]
       f = open(log, "r")
       txt=f.read()
       acc = float(txt.split("mean of 2 way clf acc full rotation =")[1].split("\nbedbench_bedchair")[0])
       return acc
    # def getAB_acc(logID='17800181',axis="AB"):

    #     log = glob(f'./logs/*{logID}*')[0]
    #     f = open(log, "r")
    #     txt=f.read()
    #     # acc = float(txt.split("bedbench_bedchair ")[1].split("\nbedbench_bedtable")[0])
    #     t = txt.split(twoWayClfDict[axis][0])
    #     accs=[]
    #     for i in range(1,9):
    #         tt=float(t[i].split(twoWayClfDict[axis][1])[0])
    #         # print(tt)
    #         accs.append(tt)
    #     return accs
    
    # ?????? AB CD AC BD???two way clf acc???
    twoWayClfDict={
        "AB":['bedtable_bedchair','\nbedtable_tablebench'],
        "AC":['bedchair_bedtable','\nbedchair_chairbench'],
        "AD":['bedchair_bedbench','\nbedchair_bedtable'],
        "BC":['bedchair_chairtable','\nbedtable_bedbench'],
        "BD":['bedchair_chairbench','\nbedchair_chairtable'],
        "CD":['bedtable_tablebench','\nbedtable_tablechair'],
    }
    def getAB_acc(logID='',axis=""):

        log = glob(f'./logs/*{logID}*')[0]
        f = open(log, "r")
        txt=f.read()
        # acc = float(txt.split("bedbench_bedchair ")[1].split("\nbedbench_bedtable")[0])
        t = txt.split(twoWayClfDict[axis][0])
        accs=[]
        for i in range(1,9):
            tt=float(t[i].split(twoWayClfDict[axis][1])[0])
            accs.append(tt)
        return accs


    def getTrace_New_greedy_everySession(axis="AB"):
        ABs=[]
        t = np.mean(getAB_acc(logID='17900710',axis=axis))
        ABs.append(t)
        t = np.mean(getAB_acc(logID='17900933',axis=axis))
        ABs.append(t)
        t = np.mean(getAB_acc(logID='17901857',axis=axis))
        ABs.append(t)
        t = np.mean(getAB_acc(logID='17905902',axis=axis))
        ABs.append(t)
        return ABs

    def getTrace_for_ses1Mask(axis="AB"):
        ABs=[]
        t = np.mean(getAB_acc(logID='17929036',axis=axis))
        ABs.append(t)
        t = np.mean(getAB_acc(logID='17760273',axis=axis))
        ABs.append(t)
        t = np.mean(getAB_acc(logID='17792316',axis=axis))
        ABs.append(t)
        t = np.mean(getAB_acc(logID='17799958',axis=axis))
        ABs.append(t)
        return ABs
    
    


    imcodeDict={
        'A': 'bed',
        'B': 'chair',
        'C': 'table',
        'D': 'bench'}
    # twoWayClfDict={
    #     "AB":['bedtable_bedchair','\nbedtable_tablebench'],
    #     "AC":['bedchair_bedtable','\nbedchair_chairbench'],
    #     "AD":['bedchair_bedbench','\nbedchair_bedtable'],
    #     "BC":['bedchair_chairtable','\nbedtable_bedbench'],
    #     "BD":['bedchair_chairbench','\nbedchair_chairtable'],
    #     "CD":['bedtable_tablebench','\nbedtable_tablechair'],
    # }
   logID={1:'00000181',2:'17800181',3:'17800357',4:'17806127',5:'17800522'}
   BestROIs=[]
   FourWayAcc=[]
   TwoWayAcc=[]
   for currSess in range(1,6):
       BestROIs.append(getBestROIs(logID=logID[currSess]))
       FourWayAcc.append(get4wayacc(logID=logID[currSess]))
       TwoWayAcc.append(get2wayacc(logID=logID[currSess]))
   BestROIs

   _bestROIs=[]
   for i in BestROIs:
       _bestROIs.append([int(a) for a in i])

   _=plt.figure(figsize=(10,10))
   for currSession in range(1,len(_bestROIs)+1):
       plt.scatter(_bestROIs[currSession-1],[currSession]*len(_bestROIs[currSession-1]),s=10,label=f"session{currSession}")
   plt.title("compare the best ROI selected from each session")   
   plt.ylabel("session")
   plt.xlabel("ROI ID")





   def inRatio(a1,a2):
       count=0
       for i in a2:
           if i in a1:
               count+=1
       return count/len(a1)
   plt.figure(figsize=(16, 8))
   for i in range(1,6):
       ratios=[]
       for currSession in range(1,6):
           t = inRatio(_bestROIs[i-1],_bestROIs[currSession-1])
           ratios.append(t)
       plt.subplot(1,6,i)
       plt.plot(np.arange(1,6),ratios)
       plt.xlabel("session")
       plt.ylim([0,1])
       plt.title(f"ROI in ses x out of ses{i}")



   array=np.zeros((5,300))
   for currROI in range(1,301):
       for currSes in range(1,6):
           if currROI in _bestROIs[currSes-1]:
               array[currSes-1, currROI-1] = 1
   # plt.imshow(array)
   plt.figure(figsize=(10,10))
   plt.plot(np.arange(1,301),np.sum(array,axis=0))
   plt.xlabel("ROI ID")
   plt.ylabel("count of existence")



   arrayMean=np.sum(array,axis=0)
   print(f"ID of survived ROI in all sessions={np.where(arrayMean==5)[0]+1},possible range 1-300")

   arrayMean=np.sum(array,axis=0)
   arrayMean[arrayMean==0]=None
   plt.hist(arrayMean)
   #plt.xlim([0.9,6.5])




   # cd /gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/subjects/sub003/ses1/recognition/mask
   # load_fsl
   # fslview_deprecated GMschaefer_8.nii.gz GMschaefer_159.nii.gz GMschaefer_160.nii.gz GMschaefer_163.nii.gz
   # 8 159 160 163

def run12_34_data_quality(cfg,testRun=None,recordingTxt=None,forceGreedy="",HeadRun_TailRun="HeadRun"):
    # To do 1: In ses2 3 4:  
    # A = acc(train on 1 test on 2)+acc(train on 2 test on 1); 
    # B = acc(train on 3 test on 4)+acc(train on 4 test on 3);
    # Compare A and B to check data quality. 

    # ?????????1??????SES2 3 4??? 
    # A = acc(??????1??????2)+acc(??????2??????1)???
    # B = acc(training on 3 test on 4)+acc(training on 4 test on 3)???
    # ??????A???B????????????????????????
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
        main_dir='/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/'
    else:
        main_dir='/Users/kailong/Desktop/rtEnv/rt-cloud/projects/rtSynth_rt/'

    working_dir=main_dir
    os.chdir(working_dir)

    '''
    if you read runRecording for current session and found that there are only 4 runs in the current session, 
    you read the runRecording for previous session and fetch the last 4 recognition runs from previous session
    '''
    runRecording = pd.read_csv(f"{cfg.recognition_dir}../runRecording.csv")
    actualRuns = list(runRecording['run'].iloc[list(np.where(1==1*(runRecording['type']=='recognition'))[0])]) # can be [1,2,3,4,5,6,7,8] or [1,2,4,5]
    if HeadRun_TailRun=='HeadRun':
        actualRuns=actualRuns[:2]
    elif HeadRun_TailRun=='TailRun':
        actualRuns=actualRuns[2:]
    print(f"actualRuns={actualRuns}")
    objects = ['bed', 'bench', 'chair', 'table']

    new_run_indexs=[]
    new_run_index=1 #????????????run ???index?????????????????????testRun selection???????????????????????????????????? new_run_index ?????????1???2???3???4???5???6???7???8
    for ii,run in enumerate(actualRuns): # load behavior and brain data for current session
        t = np.load(f"{cfg.recognition_dir}brain_run{run}.npy")
        if forceGreedy=="forceGreedy":
            mask = np.load(f"{cfg.chosenMask_using}")
            print(f"loading {cfg.chosenMask_using}")
        else:
            mask = np.load(f"{cfg.chosenMask}")
            print(f"loading {cfg.chosenMask}")
        t = t[:,mask==1]
        t = normalize(t)
        brain_data=t if ii==0 else np.concatenate((brain_data,t), axis=0)

        t = pd.read_csv(f"{cfg.recognition_dir}behav_run{run}.csv")
        t['run_num'] = new_run_index
        new_run_indexs.append(new_run_index)
        new_run_index+=1
        behav_data=t if ii==0 else pd.concat([behav_data,t])

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
        new_trained_full_rotation_4_way_accuracy_mean = np.mean(list(accList.values()))
        if recordingTxt: #if tmp_folder is not None but some string, save the sentence.
            append_file(f"{recordingTxt}",f"new trained full rotation 4 way accuracy mean={np.mean(list(accList.values()))}")
        
        return accList,new_trained_full_rotation_4_way_accuracy_mean
    accList,new_trained_full_rotation_4_way_accuracy_mean = train4wayClf(META, FEAT)
    
    # ??????full rotation???2way clf???accuracy ????????? ??????
    accs_rotation=[]
    print(f"new_run_indexs={new_run_indexs}")
    for testRun in new_run_indexs:
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
        print(f"testRun = {testRun} : average 2 way clf accuracy={np.mean(list(accs.values()))}")
        accs_rotation.append(np.mean(list(accs.values())))
    print(f"mean of 2 way clf acc full rotation = {np.mean(accs_rotation)}")
    mean_of_2_way_clf_acc_full_rotation = np.mean(accs_rotation)
    # if recordingTxt: #if tmp_folder is not None but some string, save the sentence.
    #     append_file(f"{recordingTxt}",f"mean of 2 way clf acc full rotation = {np.mean(accs_rotation)}")

    return new_trained_full_rotation_4_way_accuracy_mean , mean_of_2_way_clf_acc_full_rotation

def compareHeadRuns_and_TailRuns():
    # ??????????????????run12_34_data_quality?????????????????????????????????????????????feedback session?????????????????????recognition run????????????recognition run???data quality
    import sys
    sys.path.append('/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/expScripts/recognition/')
    from rtCommon.cfg_loading import mkdir,cfg_loading
    from recognition_dataAnalysisFunctions import normalize
    import matplotlib.pyplot as plt
    Fourway=[]
    Twoway=[]
    for feedback_ses in [2,3,4]:
        cfg = cfg_loading(f"sub003.ses{feedback_ses}.toml")
        new_trained_full_rotation_4_way_accuracy_mean , mean_of_2_way_clf_acc_full_rotation = run12_34_data_quality(cfg,HeadRun_TailRun="HeadRun")
        Fourway.append(new_trained_full_rotation_4_way_accuracy_mean)
        Twoway.append(mean_of_2_way_clf_acc_full_rotation)
        
    print(Fourway)
    print(Twoway)

    Tail_Fourway=[]
    Tail_Twoway=[]
    for feedback_ses in [2,3,4]:
        cfg = cfg_loading(f"sub003.ses{feedback_ses}.toml")
        new_trained_full_rotation_4_way_accuracy_mean , mean_of_2_way_clf_acc_full_rotation = run12_34_data_quality(cfg,HeadRun_TailRun="TailRun")
        Tail_Fourway.append(new_trained_full_rotation_4_way_accuracy_mean)
        Tail_Twoway.append(mean_of_2_way_clf_acc_full_rotation)

    print(Tail_Fourway)
    print(Tail_Twoway)


    plt.figure()
    plt.plot([2,3,4],Fourway,label="4 way HeadRuns",c='r')
    plt.plot([2,3,4],Tail_Fourway,label="4 way TailRuns",c='b')
    plt.legend()

    plt.figure()
    plt.plot([2,3,4],Twoway,label="2 way HeadRuns",c='r')
    plt.plot([2,3,4],Tail_Twoway,label="2 way TailRuns",c='b')
    plt.legend()

def organize_newGreedy_plot(): #???????????????????????????????????????ses???mask???????????????????????????ses??????????????????mask??????????????????????????????4way????????????2way????????????AB CD AC AD ????????????????????????

    # ??????ses1???leave one run out greedy mask finding????????????????????????7???run????????????????????????run??????????????????????????????run?????????????????? ??????????????????2way?????????
    def getAB_acc(logID='',axis="", twoWayClfDict=''):

        log = glob(f'./logs/*{logID}*')[0]
        f = open(log, "r")
        txt=f.read()
        # acc = float(txt.split("bedbench_bedchair ")[1].split("\nbedbench_bedtable")[0])
        t = txt.split(twoWayClfDict[axis][0])
        accs=[]
        for i in range(1,9):
            tt=float(t[i].split(twoWayClfDict[axis][1])[0])
            accs.append(tt)
        return accs

    def getAB_acc_ses1_leaveout(logID='17927715',axis="AB", twoWayClfDict=''):
        
        log = glob(f'./logs/*{logID}*')[0]
        f = open(log, "r")
        txt=f.read()
        # acc = float(txt.split("bedbench_bedchair ")[1].split("\nbedbench_bedtable")[0])
        t = txt.split(f"{twoWayClfDict[axis][0]}': ")
        acc = t[1].split(f", '{twoWayClfDict[axis][1][1:]}")[0]
        return float(acc)

    ses1Log_dict = {
            1:'17943768',
            2:'17927715',
            3:'17924439',
            4:'17924440',
            5:'17924441',
            6:'17924442',
            7:'17927934',
            8:'17924445'
        }

    AB=[]
    for CurrLeaveOut in ses1Log_dict:
        t = getAB_acc_ses1_leaveout(logID=ses1Log_dict[CurrLeaveOut],axis="AB",twoWayClfDict=twoWayClfDict)
        AB.append(t)
    CD=[]
    for CurrLeaveOut in ses1Log_dict:
        t = getAB_acc_ses1_leaveout(logID=ses1Log_dict[CurrLeaveOut],axis="CD",twoWayClfDict=twoWayClfDict)
        CD.append(t)

    AC=[]
    for CurrLeaveOut in ses1Log_dict:
        t = getAB_acc_ses1_leaveout(logID=ses1Log_dict[CurrLeaveOut],axis="AC",twoWayClfDict=twoWayClfDict)
        AC.append(t)
        
    AD=[]
    for CurrLeaveOut in ses1Log_dict:
        t = getAB_acc_ses1_leaveout(logID=ses1Log_dict[CurrLeaveOut],axis="AD",twoWayClfDict=twoWayClfDict)
        AD.append(t)

    print(f"AB={np.mean(AB)}")
    print(f"CD={np.mean(CD)}")
    print(f"AC={np.mean(AC)}")
    print(f"AD={np.mean(AD)}")

    # AB_ses1 ??? ?????????????????????????????????????????????ses???????????????????????????????????????????????????ses?????????????????????????????? acc?????????????????????
    AB_ses1=np.mean(AB)
    CD_ses1=np.mean(CD)
    AC_ses1=np.mean(AC)
    AD_ses1=np.mean(AD)



    # ??????ses1???leave one run out greedy mask finding????????????????????????7???run????????????????????????run??????????????????????????????run?????????????????? ??????????????????4way?????????
    def getAB_acc_ses1_leaveout_4way(CurrLeaveOut=1):
        
        logID=ses1Log_dict[CurrLeaveOut]
        # print(logID)
        log = glob(f'./logs/*{logID}*')[0]
        f = open(log, "r")
        txt=f.read()
        # acc = float(txt.split("bedbench_bedchair ")[1].split("\nbedbench_bedtable")[0])
        t = txt.split("acc= ")
        accs=[]
        for i in range(1,9):
            accs.append(float(t[i].split("\n")[0]))
        return accs[CurrLeaveOut-1]

    FourWay_ses1=[]
    for CurrLeaveOut in ses1Log_dict:
        # print(CurrLeaveOut)
        t = getAB_acc_ses1_leaveout_4way(CurrLeaveOut=CurrLeaveOut)
        FourWay_ses1.append(t)
    print(FourWay_ses1)

    def getAB_acc_ses1_leaveout_2way(CurrLeaveOut=1):
        logID=ses1Log_dict[CurrLeaveOut]
        # print(logID)
        log = glob(f'./logs/*{logID}*')[0]
        f = open(log, "r")
        txt=f.read()
        TwoWay_ses1 = float(txt.split("average 2 way clf accuracy=")[1].split("\n\n")[0])
        return TwoWay_ses1
    TwoWay_ses1=[]
    for CurrLeaveOut in ses1Log_dict:
        # print(CurrLeaveOut)
        t = getAB_acc_ses1_leaveout_2way(CurrLeaveOut=CurrLeaveOut)
        TwoWay_ses1.append(t)
    print(TwoWay_ses1)


    twoWayClfDict={
        "AB":['bedtable_bedchair','\nbedtable_tablebench'],
        "AC":['bedchair_bedtable','\nbedchair_chairbench'],
        "AD":['bedchair_bedbench','\nbedchair_bedtable'],
        "BC":['bedchair_chairtable','\nbedtable_bedbench'],
        "BD":['bedchair_chairbench','\nbedchair_chairtable'],
        "CD":['bedtable_tablebench','\nbedtable_tablechair'],
    }
    def getAB_acc(logID='',axis=""):

        log = glob(f'./logs/*{logID}*')[0]
        f = open(log, "r")
        txt=f.read()
        # acc = float(txt.split("bedbench_bedchair ")[1].split("\nbedbench_bedtable")[0])
        t = txt.split(twoWayClfDict[axis][0])
        accs=[]
        for i in range(1,9):
            tt=float(t[i].split(twoWayClfDict[axis][1])[0])
            accs.append(tt)
        return accs


    def getTrace_New_greedy_everySession(axis="AB"):
        ABs=[]
        t = np.mean(getAB_acc(logID='17900710',axis=axis))
        ABs.append(t)
        t = np.mean(getAB_acc(logID='17900933',axis=axis))
        ABs.append(t)
        t = np.mean(getAB_acc(logID='17901857',axis=axis))
        ABs.append(t)
        t = np.mean(getAB_acc(logID='17905902',axis=axis))
        ABs.append(t)
        return ABs
    def getTrace_for_ses1Mask(axis="AB"):
        ABs=[]
        t = np.mean(getAB_acc(logID='17932998',axis=axis))
        ABs.append(t)
        t = np.mean(getAB_acc(logID='17929036',axis=axis))
        ABs.append(t)
        t = np.mean(getAB_acc(logID='17760273',axis=axis))
        ABs.append(t)
        t = np.mean(getAB_acc(logID='17792316',axis=axis))
        ABs.append(t)
        t = np.mean(getAB_acc(logID='17799958',axis=axis))
        ABs.append(t)
        return ABs

    # ?????? AB CD???acc???2345 session
    AB = getTrace_New_greedy_everySession(axis="AB")
    CD = getTrace_New_greedy_everySession(axis="CD")

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(np.arange(2,6),AB,label="AB")
    plt.plot(np.arange(2,6),CD,label="CD")
    plt.ylabel("testing acc")
    plt.xlabel("session ID")
    plt.title("new mask")
    plt.xlim([0.8,5.2])
    plt.ylim([0.8,1])


    plt.scatter(1,AB_ses1,label='AB_ses1')
    plt.scatter(1,CD_ses1,label='CD_ses1')
    plt.legend()



    # ?????? AB CD???acc???2345 session
    AB = getTrace_for_ses1Mask(axis="AB")
    CD = getTrace_for_ses1Mask(axis="CD")
    plt.figure()
    plt.plot(np.arange(1,6),AB,label="AB")
    plt.plot(np.arange(1,6),CD,label="CD")
    plt.legend()
    plt.ylabel("testing acc")
    plt.xlabel("session ID")
    plt.title("ses1 mask")
    plt.xlim([0.8,5.2])
    plt.ylim([0.8,1])




    # Get AC AD clf accuracy across sessions to know whether we gain from chasing mask.(aka greedy every session)
    # ??????AC AD???acc ???ses2345??????
    AC = getTrace_New_greedy_everySession(axis="AC")
    AD = getTrace_New_greedy_everySession(axis="AD")
    plt.figure()
    plt.plot(np.arange(2,6),AC,label="AC")
    plt.plot(np.arange(2,6),AD,label="AD")
    plt.legend()
    plt.ylabel("testing acc")
    plt.xlabel("session ID")
    plt.title("new mask")
    plt.xlim([0.8,5.2])
    plt.ylim([0.52,1])


    plt.scatter(1,AC_ses1,label='AC_ses1')
    plt.scatter(1,AD_ses1,label='AD_ses1')
    plt.legend()



    AC = getTrace_for_ses1Mask(axis="AC")
    AD = getTrace_for_ses1Mask(axis="AD")
    plt.figure()
    plt.plot(np.arange(1,6),AC,label="AC")
    plt.plot(np.arange(1,6),AD,label="AD")
    plt.legend()
    plt.ylabel("testing acc")
    plt.xlabel("session ID")
    plt.title("ses1 mask")
    plt.xlim([0.8,5.2])
    plt.ylim([0.52,1])


    def getAB_acc(logID=''):

        log = glob(f'./logs/*{logID}*')[0]
        f = open(log, "r")
        txt=f.read()
        # acc = float(txt.split("bedbench_bedchair ")[1].split("\nbedbench_bedtable")[0])
        t = txt.split(twoWayClfDict[axis][0])
        accs=[]
        for i in range(1,9):
            tt=float(t[i].split(twoWayClfDict[axis][1])[0])
            accs.append(tt)
        return accs
    def get_2_4_way_acc(logID=''):
        log = glob(f'./logs/*{logID}*')[0]
        f = open(log, "r")
        txt=f.read()
        FourWayAcc = float(txt.split("new trained full rotation 4 way accuracy mean=")[1].split("\nnew_run_indexs")[0])

        TwoWayAcc = float(txt.split("mean of 2 way clf acc full rotation = ")[1].split("\nbedbench_bedchair")[0])
        return FourWayAcc,TwoWayAcc




    def getTrace_New_greedy_everySession_2way_4way():
        FourWayAccs,TwoWayAccs=[],[]
        everySesMaskDict={2:'17900710',3:'17900933',4:'17901857',5:'17905902'}
        for ID in everySesMaskDict:
            FourWayAcc,TwoWayAcc = get_2_4_way_acc(logID=everySesMaskDict[ID])
            FourWayAccs.append(FourWayAcc)
            TwoWayAccs.append(TwoWayAcc)
        return FourWayAccs,TwoWayAccs

    def getTrace_for_ses1Mask_2way_4way():
        FourWayAccs,TwoWayAccs=[],[]
        ses1Mask_dict = {1:'17932998',2:'17929036',3:'17760273',4:'17792316',5:'17799958'}
        for ID in ses1Mask_dict:
            FourWayAcc,TwoWayAcc = get_2_4_way_acc(logID=ses1Mask_dict[ID])
            FourWayAccs.append(FourWayAcc)
            TwoWayAccs.append(TwoWayAcc)
        return FourWayAccs,TwoWayAccs

    # ??????ses1???mask????????????4way??????2way???acc??????
    FourWayAccs_maskSes1,TwoWayAccs_maskSes1 = getTrace_for_ses1Mask_2way_4way()
    plt.figure()
    plt.plot(np.arange(1,6),FourWayAccs_maskSes1,label="FourWayAccs_maskSes1")
    plt.plot(np.arange(1,6),TwoWayAccs_maskSes1,label="TwoWayAccs_maskSes1")
    plt.legend()
    plt.title("ses1 mask")
    plt.xlabel("session ID")
    plt.ylabel("acc")
    plt.ylim([0.45,1])

    FourWayAccs_maskEverySes,TwoWayAccs_maskEverySes = getTrace_New_greedy_everySession_2way_4way()
    plt.figure()
    plt.plot(np.arange(2,6),FourWayAccs_maskEverySes,label="FourWayAccs_maskEverySes")
    plt.plot(np.arange(2,6),TwoWayAccs_maskEverySes,label="TwoWayAccs_maskEverySes")
    plt.scatter(1,np.mean(FourWay_ses1),label='FourWay_ses1')
    plt.scatter(1,np.mean(TwoWay_ses1),label='TwoWay_ses1')
    plt.legend()
    plt.title("new mask")
    plt.xlabel("session ID")
    plt.ylabel("acc")
    plt.ylim([0.45,1])

def post_pipeline():
    # sub003
    pre_pipeline_1___compare_forceGreedy_list={
        'ses2':17978038,
        'ses3':17978125,
        'ses4':17978126,
        'ses5':17978127
    }

    pre_pipeline_2___forceGreedy_list={
        'ses2':17978744,
        'ses3':17978751,
        'ses4':17978766,
        'ses5':17978787
    }

    pre_pipeline_3___ses1_LOO_Greedy_and_trainTest_list={
        'run1':17980364,
        'run2':17980365,
        'run3':17980366,
        'run4':17980368,
        'run5':17980369,
        'run6':17980370,
        'run7':17980373,
        'run8':17980374
    }






    # # sub002
    # pre_pipeline_1___compare_forceGreedy_list={
    #     'ses2':17982594,
    #     'ses3':17982614,
    #     'ses4':17982664,
    #     'ses5':17982673,
    #     'ses6':17982888
    # }


    # pre_pipeline_2___forceGreedy_list={
    #     'ses2':18033816,
    #     'ses3':18033905,
    #     'ses4':18033945,
    #     'ses5':18033989,
    #     'ses6':18034020
    # }

    # # pre_pipeline_3___ses1_LOO_Greedy_and_trainTest_list={
    # #     'run1':18034656,
    # #     'run2':18034657,
    # #     'run3':18034658,
    # # #     'run4':,
    # # #     'run5':,
    # # #     'run6':,
    # # #     'run7':,
    # # #     'run8':
    # # }

    # pre_pipeline_3___ses1_LOO_Greedy_and_trainTest_list={
    #     'run1':18034836,
    #     'run2':18034839,
    #     'run3':18034840,
    #     'run4':18034844,
    #     'run5':18034846,
    #     'run6':18034927,
    #     'run7':18034929,
    #     'run8':18035008
    # }

    import pandas as pd
    import matplotlib.pyplot as plt
    main_dir="/gpfs/milgram/project/turk-browne/projects/rt-cloud/"



    def getSummaryPlotFor(axis,offset=0.2): #axis can be 'Fourway_acc' or "AB_acc" etc
        # ?????? pre-pipeline 3 ??? pre-pipeline 2 ????????????

        # ??????ses1????????????
        Fourway_acc_ses1={}
        for curr_run,run in enumerate(pre_pipeline_3___ses1_LOO_Greedy_and_trainTest_list):
            ID = pre_pipeline_3___ses1_LOO_Greedy_and_trainTest_list[run]
            t=pd.read_csv(main_dir+f"logs/accTable_{ID}.csv")
            Fourway_acc_ses1[run]=t.loc[curr_run,axis]
        print(f"axis={axis}\nFourway_acc_ses1={Fourway_acc_ses1}\n\n")

        # ?????? ses 2345
        pre_pipeline_2___Fourway_acc_ses2345={}
        for ses in pre_pipeline_2___forceGreedy_list:
            ID = pre_pipeline_2___forceGreedy_list[ses]
            t=pd.read_csv(main_dir+f"logs/accTable_{ID}.csv")
            pre_pipeline_2___Fourway_acc_ses2345[ses] = t[axis]

        pre_pipeline_2___Fourway_acc_ses2345['ses1']=list(Fourway_acc_ses1.values())
        pre_pipeline_2___Fourway_acc_ses2345

        # ??????
        plt.figure()
        pre_pipeline_2___Fourway_acc_ses2345_mean=[]
        sesNumber=len(pre_pipeline_2___Fourway_acc_ses2345)
        for ses in range(1,sesNumber+1):
            t=pre_pipeline_2___Fourway_acc_ses2345[f'ses{ses}']
            pre_pipeline_2___Fourway_acc_ses2345_mean.append(np.mean(t))
            plt.scatter([ses]*len(t),t,c='r',s=10)
        plt.plot(np.arange(1,sesNumber+1),pre_pipeline_2___Fourway_acc_ses2345_mean,c='r',label=f"{axis} new mask")



        # ?????? pre-pipeline 3 ??? pre-pipeline 1 ????????????
        pre_pipeline_1___Fourway_acc_ses2345={}
        for ses in pre_pipeline_1___compare_forceGreedy_list:
            ID = pre_pipeline_1___compare_forceGreedy_list[ses]
            t=pd.read_csv(main_dir+f"logs/accTable_{ID}.csv")
            pre_pipeline_1___Fourway_acc_ses2345[ses]=t[axis]
        pre_pipeline_1___Fourway_acc_ses2345['ses1']=list(Fourway_acc_ses1.values())


        # ??????
        pre_pipeline_1___Fourway_acc_ses2345_mean=[]
        for ses in range(1,sesNumber+1):
            t=pre_pipeline_1___Fourway_acc_ses2345[f'ses{ses}']
            pre_pipeline_1___Fourway_acc_ses2345_mean.append(np.mean(t))
            plt.scatter([ses+offset]*len(t),t,c='b',s=10)
        plt.plot(np.arange(1,sesNumber+1),pre_pipeline_1___Fourway_acc_ses2345_mean,c='b',label=f"{axis} ses1 mask")

        plt.legend()
        plt.title(axis)
        plt.ylim(0,1)
        # ??????????????????????????????
    for axis in ['Fourway_acc','AB_acc','CD_acc','AC_acc','AD_acc','BC_acc','BD_acc']:
        getSummaryPlotFor(axis)



    def getSummaryPlotFor(axis,offset=0.2,color='b'): #axis can be 'Fourway_acc' or "AB_acc" etc
        # ?????? pre-pipeline 3 ??? pre-pipeline 2 ????????????

        # ??????ses1????????????
        Fourway_acc_ses1={}
        for curr_run,run in enumerate(pre_pipeline_3___ses1_LOO_Greedy_and_trainTest_list):
            ID = pre_pipeline_3___ses1_LOO_Greedy_and_trainTest_list[run]
            t=pd.read_csv(main_dir+f"logs/accTable_{ID}.csv")
            Fourway_acc_ses1[run]=t.loc[curr_run,axis]
        Fourway_acc_ses1

    #     # ?????? ses 2345
    #     pre_pipeline_2___Fourway_acc_ses2345={}
    #     for ses in pre_pipeline_2___forceGreedy_list:
    #         ID = pre_pipeline_2___forceGreedy_list[ses]
    #         t=pd.read_csv(main_dir+f"logs/accTable_{ID}.csv")
    #         pre_pipeline_2___Fourway_acc_ses2345[ses] = t[axis]

    #     pre_pipeline_2___Fourway_acc_ses2345['ses1']=list(Fourway_acc_ses1.values())
    #     pre_pipeline_2___Fourway_acc_ses2345

    #     # ??????
    #     pre_pipeline_2___Fourway_acc_ses2345_mean=[]
    #     sesNumber=len(pre_pipeline_2___Fourway_acc_ses2345)
    #     for ses in range(1,sesNumber+1):
    #         t=pre_pipeline_2___Fourway_acc_ses2345[f'ses{ses}']
    #         pre_pipeline_2___Fourway_acc_ses2345_mean.append(np.mean(t))
    #         plt.scatter([ses+offset]*len(t),t,c=color,s=10)
    #     plt.plot(np.arange(1,sesNumber+1),pre_pipeline_2___Fourway_acc_ses2345_mean,label=f"{axis} new mask",c=color)



        # ?????? pre-pipeline 3 ??? pre-pipeline 1 ????????????
        pre_pipeline_1___Fourway_acc_ses2345={}
        for ses in pre_pipeline_1___compare_forceGreedy_list:
            ID = pre_pipeline_1___compare_forceGreedy_list[ses]
            t=pd.read_csv(main_dir+f"logs/accTable_{ID}.csv")
            pre_pipeline_1___Fourway_acc_ses2345[ses]=t[axis]
        pre_pipeline_1___Fourway_acc_ses2345['ses1']=list(Fourway_acc_ses1.values())


        # ??????
        pre_pipeline_1___Fourway_acc_ses2345_mean=[]
        sesNumber=len(pre_pipeline_1___Fourway_acc_ses2345)
        for ses in range(1,sesNumber+1):
            t=pre_pipeline_1___Fourway_acc_ses2345[f'ses{ses}']
            pre_pipeline_1___Fourway_acc_ses2345_mean.append(np.mean(t))
            plt.scatter([ses+offset]*len(t),t,c=color,s=10)
        plt.plot(np.arange(1,sesNumber+1),pre_pipeline_1___Fourway_acc_ses2345_mean,label=f"{axis} ses1 mask",c=color)

        plt.legend()
        plt.title(axis)
        plt.ylim(0,1)
        # ??????????????????????????????

    getSummaryPlotFor('AB_acc',offset=0.0,color='b')
    getSummaryPlotFor('CD_acc',offset=0.2,color='r')
    plt.title("AB_clf and CD_clf using ses1 mask")

    getSummaryPlotFor('AC_acc',offset=0.0,color='red')
    getSummaryPlotFor('AD_acc',offset=0.1,color='orange')
    getSummaryPlotFor('BC_acc',offset=0.2,color='black')
    getSummaryPlotFor('BD_acc',offset=0.3,color='blue')

    plt.title("AC AD BC BD comparison")