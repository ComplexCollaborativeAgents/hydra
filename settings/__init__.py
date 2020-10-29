import sys
import os
from os import path

# Create your own local_settings.py file in this directory if you want
# to override this variable and not run headless
HEADLESS = True
SCREENSHOT = False
DEBUG = False
NO_PLANNING = False
SB_DEV_MODE = True

SB_PLANNER_MEMORY_LIMIT = 50 # memory limit for UPMurphi (in MB)
SB_DELTA_T = 0.05 # time discretisation for UPMurphi
SB_TIMEOUT = 30 # timeout for the planning phase (in seconds)
SB_PLANNER_SIMPLIFICATION_SEQUENCE = [0,1] # the order of problem simplications attempted to generate a plan

CP_PLANNER_MEMORY_LIMIT = 50 # memory limit for UPMurphi (in MB)
CP_DELTA_T = 0.02 # time discretisation for UPMurphi
CP_TIMEOUT = 60 # timeout for the planning phase (in seconds)

OS_ROOT_PATH = path.abspath(os.sep)
ROOT_PATH = path.join(path.dirname(path.dirname(path.abspath(__file__))))
SCIENCE_BIRDS_BIN_DIR = path.join(ROOT_PATH,'bin')
SCIENCE_BIRDS_LEVELS_DIR = path.join(SCIENCE_BIRDS_BIN_DIR,'linux','Levels','novelty_level_0','type2','Levels')
SB_INIT_COLOR_MAP = path.join(ROOT_PATH,'worlds','science_birds_interface','demo','ColorMap.json')
SB_SIM_SPEED = 30 # run at real time
SB_GT_FREQ = int(30/SB_SIM_SPEED)
SB_CLASSIFICATION_THRESHOLD = 0.5


SCIENCE_BIRDS_SERVER_CMD = 'java -jar {}'.format(path.join(SCIENCE_BIRDS_BIN_DIR, 'linux','game_playing_interface.jar'))

PLANNING_DOCKER_PATH = path.join(ROOT_PATH, 'agent', 'planning', 'docker_scripts')
CARTPOLE_PLANNING_DOCKER_PATH = path.join(ROOT_PATH, 'agent', 'planning', 'cartpole_docker_scripts')
VAL_DOCKER_PATH = path.join(ROOT_PATH, 'agent', 'planning', 'val_scripts')

sc_json_config =  [{
    "host": "127.0.0.1",
    "port": "2004",
    "requestbufbytes": 4,
    "d": 4,
    "e": 5}]

try:
    print('importing local settings')
    from .local_settings import *
except ImportError:
    print('import error!!!!')
    pass
