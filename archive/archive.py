def compareScore(cfg,testRun=None): 
    # 这个函数是从minimalClass修改过来的。目的是为了使用ses4的模型来对比ses5的前两个recognition run和后两个recognition run 的testing score
    # cfg 使用的是ses5的cfg，可以使用cfg.usingModel_dir 的模型，但是使用的是ses5的数据
    # http://localhost:8125/notebooks/projects/rtSynth_rt/archive/compareScore.ipynb
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
        main_dir='/Users/kailong/Desktop/rtEnv/rt-cloud/projects/rtSynth_rt'
    working_dir=main_dir
    os.chdir(working_dir)

    '''
    if you read runRecording for current session and found that there are only 4 runs in the current session, 
    you read the runRecording for previous session and fetch the last 4 recognition runs from previous session
    '''
    runRecording = pd.read_csv(f"{cfg.recognition_dir}../runRecording.csv")
    actualRuns = list(runRecording['run'].iloc[list(np.where(1==1*(runRecording['type']=='recognition'))[0])]) # can be [1,2,3,4,5,6,7,8] or [1,2,4,5]
    
    objects = ['bed', 'bench', 'chair', 'table']

    new_run_indexs=[]
    new_run_index=1 #使用新的run 的index，以便于后面的testRun selection的时候不会重复。正常的话 new_run_index 应该是1，2，3，4，5，6，7，8
    for ii,run in enumerate(actualRuns): # load behavior and brain data for current session
        t = np.load(f"{cfg.recognition_dir}brain_run{run}.npy")
        mask = np.load(f"{cfg.chosenMask}")
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
    META=behav_data

    # convert item colume to label colume
    imcodeDict={
    'A': 'bed',
    'B': 'chair',
    'C': 'table',
    'D': 'bench'}

    # accTable = pd.DataFrame(columns=['AB_acc', 'CD_acc', 'AC_acc', 'AD_acc', 'BC_acc', 'BD_acc'])
    accTable = pd.DataFrame()

    # accTable = accTable.append({
    #     'testRun':testRun,
    #     'AB_acc':AB_acc, 
    #     'CD_acc':CD_acc, 
    #     'AC_acc':AC_acc, 
    #     'AD_acc':AD_acc, 
    #     'BC_acc':BC_acc, 
    #     'BD_acc':BD_acc},
    #     ignore_index=True)

    # accTable.to_csv(f"./logs/accTable_{cfg.jobID}.csv") 


    label=[]
    for curr_trial in range(META.shape[0]):
        label.append(imcodeDict[META['Item'].iloc[curr_trial]])
    META['label']=label # merge the label column with the data dataframe

    # Which run to use as test data (leave as None to not have test data)
    # testRun = 0 # when testing: testRun = 2 ; META['run_num'].iloc[:5]=2
    def train4wayClf(META, FEAT,accTable):
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
            Fourway_acc=acc
            accTable = accTable.append({
                'testRun':testRun,
                'Fourway_acc':Fourway_acc},
                ignore_index=True)

        print(f"new trained full rotation 4 way accuracy mean={np.mean(list(accList.values()))}")
        
        return accList,accTable
    # accList = train4wayClf(META, FEAT)
    
    # 获得full rotation的2way clf的accuracy 平均值 中文
    accs_rotation=[]
    print(f"new_run_indexs={new_run_indexs}")
    # for testRun in new_run_indexs:
    #     allpairs = itertools.combinations(objects,2)
    #     accs={}
    #     # Iterate over all the possible target pairs of objects
    #     for pair in allpairs:
    #         # Find the control (remaining) objects for this pair
    #         altpair = other(pair)
            
    #         # pull sorted indices for each of the critical objects, in order of importance (low to high)
    #         # inds = get_inds(FEAT, META, pair, testRun=testRun)
            
    #         # Find the number of voxels that will be left given your inclusion parameter above
    #         # nvox = red_vox(FEAT.shape[1], include)
            
    #         for obj in pair:
    #             # foil = [i for i in pair if i != obj][0]
    #             for altobj in altpair:
    #                 # establish a naming convention where it is $TARGET_$CLASSIFICATION
    #                 # Target is the NF pair (e.g. bed/bench)
    #                 # Classificationis is btw one of the targets, and a control (e.g. bed/chair, or bed/table, NOT bed/bench)
    #                 naming = '{}{}_{}{}'.format(pair[0], pair[1], obj, altobj)

    #                 if testRun:
    #                     trainIX = ((META['label']==obj) | (META['label']==altobj)) & (META['run_num']!=int(testRun))
    #                     testIX = ((META['label']==obj) | (META['label']==altobj)) & (META['run_num']==int(testRun))
    #                 else:
    #                     trainIX = ((META['label']==obj) | (META['label']==altobj))
    #                     testIX = ((META['label']==obj) | (META['label']==altobj))

    #                 # pull training and test data
    #                 trainX = FEAT[trainIX]
    #                 testX = FEAT[testIX]
    #                 trainY = META.iloc[np.asarray(trainIX)].label
    #                 testY = META.iloc[np.asarray(testIX)].label

    #                 assert len(np.unique(trainY))==2

    #                 # Train your classifier
    #                 clf = LogisticRegression(penalty='l2',C=1, solver='lbfgs', max_iter=1000, 
    #                                             multi_class='multinomial').fit(trainX, trainY)
                    
    #                 model_folder = cfg.trainingModel_dir
    #                 # Save it for later use
    #                 # joblib.dump(clf, model_folder +'/{}.joblib'.format(naming))
                    
    #                 # Monitor progress by printing accuracy (only useful if you're running a test set)
    #                 acc = clf.score(testX, testY)
    #                 print(naming, acc)
    #                 accs[naming]=acc
    #     print(f"testRun = {testRun} : average 2 way clf accuracy={np.mean(list(accs.values()))}")
    #     accs_rotation.append(np.mean(list(accs.values())))
    # print(f"mean of 2 way clf acc full rotation = {np.mean(accs_rotation)}")

    # 用所有数据训练要保存并且使用的模型：
    allpairs = itertools.combinations(objects,2)
    accs={}
    # Iterate over all the possible target pairs of objects
    for pair in allpairs:
        # Find the control (remaining) objects for this pair
        altpair = other(pair)
        for obj in pair:
            # foil = [i for i in pair if i != obj][0]
            for altobj in altpair:
                # establish a naming convention where it is $TARGET_$CLASSIFICATION
                # Target is the NF pair (e.g. bed/bench)
                # Classificationis is btw one of the targets, and a control (e.g. bed/chair, or bed/table, NOT bed/bench)
                naming = '{}{}_{}{}'.format(pair[0], pair[1], obj, altobj)

                trainIX = ((META['label']==obj) | (META['label']==altobj))
                testIX = ((META['label']==obj) | (META['label']==altobj))

                # pull training and test data
                trainX = FEAT[trainIX]
                testX = FEAT[testIX]
                trainY = META.iloc[np.asarray(trainIX)].label
                testY = META.iloc[np.asarray(testIX)].label

                assert len(np.unique(trainY))==2

                # Train your classifier
                # clf = LogisticRegression(penalty='l2',C=1, solver='lbfgs', max_iter=1000, 
                                            # multi_class='multinomial').fit(trainX, trainY)
                
                # Save it for later use
                clf = joblib.load(clf, cfg.usingModel_dir +'/{}.joblib'.format(naming))
                
                # Monitor progress by printing accuracy (only useful if you're running a test set)
                acc = clf.score(testX, testY)
                print(naming, acc)
                accs[naming]=acc
    print(f"average 2 way clf accuracy={np.mean(list(accs.values()))}")
    
    accTable.to_csv(f"./logs/accTable_{cfg.jobID}.csv") 

    return accs