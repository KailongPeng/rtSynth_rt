import os
import sys
import logging
import argparse
# import project modules
# Add base project path (two directories up)
currPath = os.path.dirname(os.path.realpath(__file__))
rootPath = os.path.dirname(os.path.dirname(currPath))
sys.path.append(rootPath)
from rtCommon.utils import loadConfigFile, installLoggers
from rtCommon.structDict import StructDict
from rtCommon.projectInterface import Web

# HERE: Set the path to the fMRI Python script to run here
scriptToRun = 'projects/rtSynth_rt/rtSynth_rt.py'
initScript = 'projects/rtSynth_rt/initialize.py'
finalizeScript = 'projects/rtSynth_rt/finalize.py'
# defaultConfig = 'sample.toml'
defaultConfig = "sub001.ses2.toml"

if __name__ == "__main__":
    installLoggers(logging.INFO, logging.INFO, filename=os.path.join(currPath, 'logs/sample.log'))
    argParser = argparse.ArgumentParser()
    argParser.add_argument('--filesremote', '-x', default=False, action='store_true',
                           help='dicom files retrieved from remote server')
    argParser.add_argument('--config', '-c', default='/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/projects/rtSynth_rt/conf/'+defaultConfig, type=str,
                           help='experiment file (.json or .toml)')
    argParser.add_argument('--test', '-t', default=False, action='store_true',
                           help='start projectInterface in test mode, unsecure')
    args = argParser.parse_args()

    params = StructDict({'fmriPyScript': scriptToRun,
                         'initScript': initScript,
                         'finalizeScript': finalizeScript,
                         'filesremote': args.filesremote,
                         })

    web = Web()
    web.start(params, args.config, testMode=args.test)
