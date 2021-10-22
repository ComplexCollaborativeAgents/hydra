import sys
import os
from os import path
import shortuuid

# Create your own local_settings.py file in this directory if you want
# to override this variable and not run headless
HEADLESS = True
SCREENSHOT = False
DEBUG = False
NO_PLANNING = False
NO_REPAIR = False
NO_PDDL_CONSISTENCY = False
SB_DEV_MODE = False
NOVELTY_POSSIBLE = True

SB_PLANNER_MEMORY_LIMIT = 50 # memory limit for UPMurphi (in MB)
SB_DELTA_T = 0.05 # time discretisation for UPMurphi
SB_TIMEOUT = 30 # timeout for the planning phase (in seconds)
SB_DEFAULT_SHOT = 'RANDOM_PIG'
SB_PLANNER_SIMPLIFICATION_SEQUENCE = [1,2] # the order of problem simplications attempted to generate a plan


CP_PLANNER_MEMORY_LIMIT = 50 # memory limit for UPMurphi (in MB)
CP_DELTA_T = 0.02 # time discretisation for UPMurphi
CP_TIMEOUT = 60 # timeout for the planning phase (in seconds)
CP_CONSISTENCY_THRESHOLD = 0.01

OS_ROOT_PATH = path.abspath(os.sep)
ROOT_PATH = path.join(path.dirname(path.dirname(path.abspath(__file__))))
SCIENCE_BIRDS_BIN_DIR = path.join(ROOT_PATH,'bin')
SCIENCE_BIRDS_LEVELS_DIR = path.join(SCIENCE_BIRDS_BIN_DIR,'linux','Levels','novelty_level_0','type2','Levels')
POLYCRAFT_DIR = path.join(ROOT_PATH, 'bin', 'pal', 'PolycraftAIGym')
SB_INIT_COLOR_MAP = path.join(ROOT_PATH,'worlds','science_birds_interface','demo','ColorMap.json')
SB_SIM_SPEED = 30 # run at real time
SB_GT_FREQ = int(30/SB_SIM_SPEED)
# SB_GT_FREQ = 1
SB_CLASSIFICATION_THRESHOLD = 0.35
SB_N_FRAMES = 100

HYDRA_MODEL_REVISION_ATTEMPTS = 5

# Repair parameters for ScienceBirds
SB_REPAIR_TIMEOUT = 180
SB_REPAIR_MAX_ITERATIONS = 30
SB_CONSISTENCY_THRESHOLD = 50
SB_ANOMOLY_DETECTOR_THRESHOLD = 0.55

SCIENCE_BIRDS_SERVER_CMD = 'java -jar {}'.format(path.join(SCIENCE_BIRDS_BIN_DIR, 'linux', 'game_playing_interface.jar'))
POLYCRAFT_SERVER_CMD = "./gradlew --no-daemon --stacktrace runclient"   # Must be run in pal/PolycraftAIGym
POLYCRAFT_HEADLESS = "xvfb-run -s '-screen 0 1280x1024x24'" # Prepend to Polycraft run command to run headless

SB_PLANNING_DOCKER_PATH = path.join(ROOT_PATH, 'agent', 'planning', 'sb_planning')
CARTPOLE_PLANNING_DOCKER_PATH = path.join(ROOT_PATH, 'agent', 'planning', 'cartpole_planning')
CARTPOLEPLUSPLUS_PLANNING_DOCKER_PATH = path.join(ROOT_PATH, 'agent', 'planning', 'cartpoleplusplus_planning')
VAL_DOCKER_PATH = path.join(ROOT_PATH, 'agent', 'planning', 'val_scripts')

HYDRA_INSTANCE_ID = shortuuid.uuid()
TMP_FOLDER = path.join(os.sep, 'tmp')

sc_json_config = [{
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
