# This code should be run in console room computer to display the feedback morphings
from __future__ import print_function, division
import traceback,time
# try:
import os
if 'watts' in os.getcwd():
    main_dir = "/home/watts/Desktop/ntblab/kailong/rt-cloud/projects/rtSynth_rt/"
else:
    main_dir="/Users/kailong/Desktop/rtEnv/rt-cloud/projects/rtSynth_rt/"
import sys
sys.path.append(main_dir)
sys.path.append(main_dir+"expScripts/feedback/")
sys.path.append(main_dir+'../../')
from psychopy import visual, event, core, logging, gui, data, monitors
from psychopy.hardware.emulator import launchScan, SyncGenerator
from PIL import Image
import string
import fmrisim as sim
import numpy as np
import pandas as pd
import pylink   
from tqdm import tqdm
import re
import logging
import threading
import argparse
alpha = string.ascii_uppercase
from rtCommon.subjectInterface import SubjectInterface
from rtCommon.wsRemoteService import WsRemoteService, parseConnectionArgs
from rtCommon.utils import installLoggers
from cfg_loading import mkdir,cfg_loading
sys.path.append(f'{main_dir}expScripts/recognition/')
from recognition_dataAnalysisFunctions import AdaptiveThreshold

class SubjectService:
    def __init__(self, args, webSocketChannelName='wsSubject'):
        """
        Uses the WsRemoteService framework to parse connection-related args and establish
        a connection to a remote projectServer. Instantiates a local version of
        SubjectInterface to handle client requests coming from the projectServer connection.
        Args:
            args: Argparse args related to connecting to the remote server. These include
                "-s <server>", "-u <username>", "-p <password>", "--test",
                "-i <retry-connection-interval>"
            webSocketChannelName: The websocket url extension used to connecy and communicate
                to the remote projectServer, 'wsSubject' will connect to 'ws://server:port/wsSubject'
        """
        self.subjectInterface = SubjectInterface(subjectRemote=False)
        self.wsRemoteService = WsRemoteService(args, webSocketChannelName)
        self.wsRemoteService.addHandlerClass(SubjectInterface, self.subjectInterface)

    def runDetached(self):
        """Starts the receiver in it's own thread."""
        self.recvThread = threading.Thread(name='recvThread',
                                        target=self.wsRemoteService.runForever)
        self.recvThread.setDaemon(True)
        self.recvThread.start()

def moneySummary(df):
    sessionList=np.unique(list(df['session']))
    money_total=0
    for sessionID in sessionList:
        t = df[df['session']==sessionID]
        t_money=np.sum(10*t['monetaryReward10cent'])+np.sum(5*t['monetaryReward5cent'])
        money_total+=t_money
        print(f"money for session{sessionID+1} is {int(t_money)} cents")
    print(f"total money earned is {int(money_total)} cents")

    # get a list of money for each run
    run_money=[]
    for currRunID in range(len(df)):
        run_money.append(10*df.loc[currRunID,'monetaryReward10cent']+5*df.loc[currRunID,'monetaryReward5cent'])

    # df['run_money']=run_money
    print(f"money for each run={run_money}")

argParser = argparse.ArgumentParser()

argParser.add_argument('-c', '--config', action="store", dest="config", default='sub001.ses2.toml', type=str, help='experiment file (.json or .toml)')
argParser.add_argument('-r', '--run', action="store", dest="run", default='1', type=str, help='current run')
argParser.add_argument('-s', action="store", dest="server", default="localhost:7777",
                    help="Server Address with Port [server:port]")
argParser.add_argument('-i', action="store", dest="interval", type=int, default=5,
                    help="Retry connection interval (seconds)")
argParser.add_argument('-u', '--username', action="store", dest="username", default='kp578',
                    help="rtcloud website username")
argParser.add_argument('-p', '--password', action="store", dest="password", default='kp578',
                    help="rtcloud website password")
argParser.add_argument('--test', default=False, action='store_true',
                    help='Use unsecure non-encrypted connection')
argParser.add_argument('--trying', default=False, action='store_true',
                    help='Use unsecure non-encrypted connection')
args = argParser.parse_args()


if args.trying:
    scanmode = 'Test'  # 'Scan' or 'Test' or None
    screenmode = False  # fullscr True or False
    monitor_name = "testMonitor" #"testMonitor"
else:
    scanmode = 'Scan'  # 'Scan' or 'Test' or None
    screenmode = True  # fullscr True or False
    monitor_name = "scanner"

if not re.match(r'.*:\d+', args.server):
    print("Error: Expecting server address in the form <servername:port>")
    argParser.print_help()
    sys.exit()

# Check if the ssl certificate is valid for this server address
from rtCommon.projectUtils import login, certFile, checkSSLCertAltName, makeSSLCertFile
addr, _ = args.server.split(':')
if checkSSLCertAltName(certFile, addr) is False:
    # Addr not listed in sslCert, recreate ssl Cert
    makeSSLCertFile(addr)

if args.trying:
    cfg = cfg_loading(args.config,trying="trying")
else:
    cfg = cfg_loading(args.config)
sub = cfg.subjectName
run = int(args.run)
cfg.run = run
sess = int(cfg.session)

cfg.feedback_expScripts_dir = f"{cfg.projectDir}expScripts/feedback/"

gui = True if screenmode == False else False
scnWidth, scnHeight = monitors.Monitor(monitor_name).getSizePix()
frameTolerance = 0.001  # how close to onset before 'same' frame
TRduration=int(cfg.TR)
print(f"TRduration={TRduration}")

mywin = visual.Window(
    size=[scnWidth - 100, scnHeight - 100], fullscr=screenmode, screen=1,
    winType='pyglet', allowGUI=False, allowStencil=False,
    monitor=monitor_name, color=[0,0,0], colorSpace='rgb', #color=[0,0,0]
    blendMode='avg', useFBO=True,
    units='height')

try:
    ThresholdLog=pd.read_csv(cfg.adaptiveThreshold)
except Exception as e:
    print(f"error: {e}")
    ThresholdLog = pd.DataFrame(columns=['sub', 'session', 'run', 'threshold', 'successful_trials', 'perfect_trials','monetaryReward10cent','monetaryReward5cent','monetaryReward0cent'])

ThresholdLog = AdaptiveThreshold(cfg,ThresholdLog)
ThresholdLog.to_csv(cfg.adaptiveThreshold, index=False)
print(f"ThresholdLog = \n{ThresholdLog[['run','threshold','successful_trials']].to_string(index=False)}")

threshold = ThresholdLog['threshold'].iloc[-1]
print(f"threshold={threshold}")

# similation specific
step=3 #in simulation, how quickly the morph changes ramp up. Note this is only for simulation, has nothing to do with real experiment

# trial_list designing parameters
parameterRange=[1,5,9,13] #np.arange(1,prange) #for saving time for now. np.arange(1,20) #define the range for possible parameters for preloading images. Preloading images is to make the morphing smooth during feedback
tune=4 # this parameter controls how much to morph (how strong the morphing is) (used in preloading function), tune can range from (1,6.15] when paremeterrange is np.arange(1,20)

TrialNumber=cfg.TrialNumber # how many trials are required #test trial ,each trial is 14s, 10 trials are 140s.
if args.trying:
    # TrialNumber=1
    # print("TrialNumber=1")
    pass    

## - design the trial list: the sequence of the different types of components: 
## - e.g: ITI + waiting for fMRI signal + feedback (receive model output from feedbackReceiver.py)
trial_list = pd.DataFrame(columns=['Trial','time','TR','state','newWobble'])
curTime=0
curTR=0
state=''
trial_list.append({'Trial':None,
                    'time':None,
                    'TR':None,
                    'state':None,
                    'newWobble':None},
                    ignore_index=True)

for currTrial in range(1,1+TrialNumber):

    # ITI
    for i in range(6): # should be 6TR=12s
        trial_list=trial_list.append({'Trial':currTrial,
                                    'time':curTime,
                                    'TR':curTR,
                                    'state':'ITI',
                                    'newWobble':0},
                                    ignore_index=True)
        curTime=curTime+TRduration
        curTR=curTR+1

    # waiting for metric calculation
    for i in range(3): # should be 3TR=6s
        trial_list=trial_list.append({'Trial':currTrial,
                                    'time':curTime,
                                    'TR':curTR,
                                    'state':'waiting',
                                    'newWobble':1},
                                    ignore_index=True)
        curTime=curTime+TRduration
        curTR=curTR+1
    
    # feedback trial: try minimize the whobbling
    for i in range(5): #5TR=10s
        trial_list=trial_list.append({'Trial':currTrial,
                                    'time':curTime,
                                    'TR':curTR,
                                    'state':'feedback',
                                    'newWobble':1},
                                    ignore_index=True)
        curTime=curTime+TRduration
        curTR=curTR+1

# ITI
for i in range(6): # should be 6TR=12s
    trial_list=trial_list.append({'Trial':currTrial,
                                'time':curTime,
                                'TR':curTR,
                                'state':'ITI',
                                'newWobble':0},
                                ignore_index=True)
    curTime=curTime+TRduration
    curTR=curTR+1

def sample(L,num=10):
    # This functional uniformly sample the list to be num points
    # e.g, if L is 0-99, num is 10, newList would be [9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
    # e.g, if L is 0-95, num is 10, newList would be [8, 18, 27, 37, 47, 56, 66, 75, 85, 95]
    # e.g, if L is 0-5, num is 10, newList would be [0, 0, 0, 1, 2, 2, 3, 3, 4, 5]
    sampleStep=len(L)/num 
    newList=[]
    for i in range(1,num):
        newList.append(L[int(i*sampleStep-1)])
    newList.append(L[-1])
    return newList

def countITI(states):
    c=0
    for i in states:
        if i == "ITI":
            c+=1
        else:
            return c
    return c

# check if data for this subject and run already exist, and raise an error if they do (prevent overwriting)
# newfile = cfg.feedback_dir+"{}_{}.csv".format(str(sub), str(run))
# if os.path.exists(newfile):
#     print(f'{newfile} exists')
#     raise Exception(f'{newfile} exists')

message = visual.TextStim(mywin, text=f'Waiting...',pos=(0, 0), depth=-5.0, height=0.05,units='pix')
def display(text,message): #endMorphing can be [1,5,9,13]
    message.setAutoDraw(False)
    message = visual.TextStim(mywin, text=f'{text}',pos=(0, 0), depth=-5.0, height=0.05,units='norm')
    message.setAutoDraw(True)
    return message
monetaryReward = visual.TextStim(mywin, text=f'',pos=(0, 0), depth=-5.0, height=0.05,units='norm')    
def display_monetaryReward(text,monetaryReward): #endMorphing can be [1,5,9,13]
    monetaryReward.setAutoDraw(False)
    monetaryReward = visual.TextStim(mywin, text=f'{text}',pos=(0,-0.28), depth=-5.0, #pos=(0,-120), height=25, units='pix'
                                    height=0.05,units='norm', #norm
                                    color=(0, 1, 0), colorSpace='rgb') #green color
    monetaryReward.setAutoDraw(True)
    return monetaryReward
monetaryReward.setAutoDraw(False)
emoji1 = visual.ImageStim(
    win=mywin,
    name='emoji1',
    image=cfg.feedback_expScripts_dir + './emoji1.png', mask=None,
    ori=0, pos=(0, 0), size=(0.25, 0.25),
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-5.0)
emoji5 = visual.ImageStim(
    win=mywin,
    name='emoji5',
    image=cfg.feedback_expScripts_dir + './emoji5.png', mask=None,
    ori=0, pos=(0, 0), size=(0.25, 0.25),
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-5.0)
emoji9 = visual.ImageStim(
    win=mywin,
    name='emoji9',
    image=cfg.feedback_expScripts_dir + './emoji9.png', mask=None,
    ori=0, pos=(0, 0), size=(0.25, 0.25),
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-5.0)
emoji13 = visual.ImageStim(
    win=mywin,
    name='emoji13',
    image=cfg.feedback_expScripts_dir + './emoji13.png', mask=None,
    ori=0, pos=(0, 0), size=(0.25, 0.25),
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-5.0)
def emoji(endMorphing): #endMorphing can be [1,5,9,13]
    if endMorphing ==1:
        emoji1.setAutoDraw(True)
        emoji5.setAutoDraw(False)
        emoji9.setAutoDraw(False)
        emoji13.setAutoDraw(False)
    elif endMorphing ==5:
        emoji1.setAutoDraw(False)
        emoji5.setAutoDraw(True)
        emoji9.setAutoDraw(False)
        emoji13.setAutoDraw(False)
    elif endMorphing ==9:
        emoji1.setAutoDraw(False)
        emoji5.setAutoDraw(False)
        emoji9.setAutoDraw(True)
        emoji13.setAutoDraw(False)
    elif endMorphing ==13:
        emoji1.setAutoDraw(False)
        emoji5.setAutoDraw(False)
        emoji9.setAutoDraw(False)
        emoji13.setAutoDraw(True)
    elif endMorphing == "OFF":
        emoji1.setAutoDraw(False)
        emoji5.setAutoDraw(False)
        emoji9.setAutoDraw(False)
        emoji13.setAutoDraw(False)
emoji("OFF")

# monetaryReward1 = 0
# monetaryReward5 = 0
# monetaryReward9 = 0
# monetaryReward13 = 0
monetaryReward10cent = 0
monetaryReward5cent = 0
monetaryReward0cents = 0
# preload image list for parameter from 1 to 19.
# def preloadimages(parameterRange=np.arange(1,20),tune=1):
def preloadimages(parameterRange=[1,5,9,13],tune=1):
    '''
    purpose:
        preload images into image object sequences corrresponding too each parameter
        each parameter corresponds to 40 image objects
    steps:

    '''
    tune=tune-1
    start = time.time()
    imageLists={}
    numberOfUpdates=16 # corresponds to 66 updates    
    last_image=''
    for currParameter in tqdm(parameterRange): #49
        images=[]
        print('maximum morph=',round((tune*currParameter*numberOfUpdates+2)/numberOfUpdates+1))
        for axis in ['bedTable', 'benchBed']:
            tmp_images=[]
            for currImg in range(1,int(round(tune*currParameter*numberOfUpdates+2)),int((currParameter*numberOfUpdates+2)/numberOfUpdates)):
                currMorph=100-round(currImg/numberOfUpdates+1) if axis=='benchBed' else round(currImg/numberOfUpdates+1)
                if currMorph<1 or currMorph>99:
                    raise Exception('morphing outside limit')
                curr_image=cfg.feedback_expScripts_dir+'carchair_exp_feedback/{}_{}_{}.png'.format(axis,currMorph,5)
                if curr_image!=last_image:
                    currimage=visual.ImageStim(win=mywin,
                                                name='image',
                                                image=cfg.feedback_expScripts_dir+'carchair_exp_feedback/{}_{}_{}.png'.format(axis,currMorph,5), mask=None,
                                                ori=0, pos=(0, 0), size=(0.5, 0.5),
                                                color=[1,1,1], colorSpace='rgb', opacity=1,
                                                flipHoriz=False, flipVert=False,
                                                texRes=128, interpolate=True, depth=-4.0)
                tmp_images.append(currimage)
                last_image=cfg.feedback_expScripts_dir+'carchair_exp_feedback/{}_{}_{}.png'.format(axis,currMorph,5)
            images=images+sample(tmp_images)
            tmp_images=[]
            for currImg in reversed(range(1,int(round(tune*currParameter*numberOfUpdates+1)),int((currParameter*numberOfUpdates+2)/numberOfUpdates))):
                currMorph=100-round(currImg/numberOfUpdates+1) if axis=='benchBed' else round(currImg/numberOfUpdates+1)
                curr_image=cfg.feedback_expScripts_dir+'carchair_exp_feedback/{}_{}_{}.png'.format(axis,currMorph,5)
                if curr_image!=last_image:
                    currimage=visual.ImageStim(win=mywin,
                                                name='image',
                                                image=cfg.feedback_expScripts_dir+'carchair_exp_feedback/{}_{}_{}.png'.format(axis,currMorph,5), mask=None,
                                                ori=0, pos=(0, 0), size=(0.5, 0.5),
                                                color=[1,1,1], colorSpace='rgb', opacity=1,
                                                flipHoriz=False, flipVert=False,
                                                texRes=128, interpolate=True, depth=-4.0)
                tmp_images.append(currimage)
                last_image=cfg.feedback_expScripts_dir+'carchair_exp_feedback/{}_{}_{}.png'.format(axis,currMorph,5)
            images=images+sample(tmp_images)
        imageLists.update( {currParameter : images} )
    end = time.time()
    # print("preload image duration=", end - start)
    return imageLists

_=time.time()
# imageLists=preloadimages(parameterRange=parameterRange,tune=tune)

maxTR=int(trial_list['TR'].iloc[-1])+6
# Settings for MRI sequence
MR_settings = {'TR': TRduration, 'volumes': maxTR, 'sync': 5, 'skip': 0, 'sound': True} #{'TR': 2.000, 'volumes': maxTR, 'sync': 5, 'skip': 0, 'sound': True}


# create empty dataframe to accumulate data
data = pd.DataFrame(columns=['Sub', 'Run', 'TR', 'time'])

# Create the fixation dot, and initialize as white fill.
fix = visual.Circle(mywin, units='deg', radius=0.05, pos=(0, 0), fillColor='white',
                    lineColor='black', lineWidth=0.5, opacity=0.5, edges=128)

# start global clock and fMRI pulses (start simulated or wait for real)
print('Starting sub {} in run #{}'.format(sub, run))

vol = launchScan(mywin, MR_settings, simResponses=None, mode=scanmode,
                esc_key='escape', instr='select Scan or Test, press enter',
                wait_msg='waiting for scanner...', wait_timeout=300, log=True)

backgroundImage = visual.ImageStim(
    win=mywin,
    name='image',
    image=cfg.feedback_expScripts_dir+'./carchair_exp_feedback/greyBackground.png', mask=None,
    ori=0, pos=(0, 0), size=(0.5, 0.5),
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-4.0)

# trialClock is reset in each trial to change image every TR (2s), time for each image is 2/numOfImages
trialClock = core.Clock()

# trialClock.add(10)  # initialize as a big enough number to avoid text being shown at the first time.
TR=list(trial_list['TR'])
states=list(trial_list['state'])
newWobble=list(trial_list['newWobble'])

ParameterUpdateDuration=np.diff(np.where(trial_list['newWobble']==1))[0][0]*TRduration
curr_parameter=0
remainImageNumber=[]

installLoggers(logging.INFO, logging.INFO, filename=f'{cfg.feedback_dir}SubjectService_{run}_{sess}.log')

subjectService = SubjectService(args)
subjectService.runDetached()
global CurrBestParameter, parameter, points
points=0
CurrBestParameter=19
history=pd.DataFrame(columns=['TR_scanner', 'morphParam', 'states',"B_prob","TR_milgram"])
default_parameter=19

initialMorphParam=13
morphParam=initialMorphParam
perfect_trials=0
successful_trials=0
successful_TR=0
ITIFlag=1
countdown=12
# imagePaths13=imageLists[13]
# eachTime13=ParameterUpdateDuration/len(imagePaths13)

# # tryKailong
# money=monetaryReward1*15 + monetaryReward5*10 + monetaryReward9*5 + monetaryReward13*0
# message=display(f"You have {successful_trials} successful trials in this run \n You just earned {money} cents.",message)
# time.sleep(5)

# calculated how long each image should last.
# eachTime=ParameterUpdateDuration/len(imagePaths13)
#eachTime????????????morphing frame??????????????? ??????
B_prob=0
morphParam=None
runId,trID,value,timestamp=None,None,None,None
# imagePaths=imagePaths13
# curr_parameter=len(parameters['value'])-1

time_list=[]
t_last=time.time()
while len(TR)>1: #globalClock.getTime() <= (MR_settings['volumes'] * MR_settings['TR']) + 3:
    # trialTime = trialClock.getTime()
    # keys = event.getKeys(["5","0"])  # check for triggers
    try:
        feedbackMsg = subjectService.subjectInterface.msgQueue.get(block=True, timeout=0.0001) # from subjInterface.setResult(runNum, int(this_TR), B_prob)
        runId,trID,value,timestamp = feedbackMsg.get('runId'),feedbackMsg.get('trId'),feedbackMsg.get('value'),feedbackMsg.get('timestamp')
        B_prob = float(value)
        print(f"trID={trID}")
        t=time.time()
        time_list.append(t-t_last)
        t_last=time.time()
        print(f"time_list={time_list}")
        np.save('/Users/kailong/Desktop/rtEnv/rt-cloud/projects/rtSynth_rt/subjects/sub003/ses6/feedback/time_list',np.asarray(time_list))
    except Exception as e:
        # print(f"error {e}")
        pass
