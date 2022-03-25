import sys
import os
from os import path
import shortuuid

# Bad global variables for debugging
active_bird_id_string = ''

# Create your own local_settings.py file in this directory if you want
# to override this variable and not run headless
HEADLESS = False
SCREENSHOT = False
DEBUG = False
NO_PLANNING = False
NO_REPAIR = True
NO_PDDL_CONSISTENCY = False
SB_DEV_MODE = False
NOVELTY_POSSIBLE = True


SB_ALGO_STRING = 'gbfs'
SB_HEURISTIC_STRING = '11'

SB_PLANNER_MEMORY_LIMIT = 50  # memory limit for UPMurphi (in MB)
SB_DELTA_T = 0.025  # time discretisation for UPMurphi
SB_TIMEOUT = 30  # timeout for the planning phase (in seconds)
SB_DEFAULT_SHOT = 'RANDOM_PIG'
SB_PLANNER_SIMPLIFICATION_SEQUENCE = [1]  # the order of problem simplications attempted to generate a plan
SB_COLLECT_PERCEPTION_DATA = False


CP_PLANNER_MEMORY_LIMIT = 50  # memory limit for UPMurphi (in MB)
CP_DELTA_T = 0.02  # time discretisation for UPMurphi
CP_TIMEOUT = 60  # timeout for the planning phase (in seconds)
CP_CONSISTENCY_THRESHOLD = 0.01
CP_EPISODE_TIME_LIMIT = 1200
CP_REPAIR_TIMEOUT = 180

POLYCRAFT_DELTA_T = 1 # Time discretization
POLYCRAFT_TIMEOUT = 300 # timeout for the planning phase (in seconds)

OS_ROOT_PATH = path.abspath(os.sep)
ROOT_PATH = path.join(path.dirname(path.dirname(path.abspath(__file__))))
SCIENCE_BIRDS_BIN_DIR = path.join(ROOT_PATH,'bin')
SCIENCE_BIRDS_LEVELS_DIR = path.join(SCIENCE_BIRDS_BIN_DIR, 'linux', 'Levels', 'novelty_level_0', 'type2', 'Levels')
POLYCRAFT_DIR = path.join(ROOT_PATH, 'bin', 'pal')
SB_INIT_COLOR_MAP = path.join(ROOT_PATH, 'worlds', 'science_birds_interface', 'demo', 'ColorMap.json')
SB_SIM_SPEED = 30  # run at real time
SB_GT_FREQ = 30  # int(30/SB_SIM_SPEED)
# SB_GT_FREQ = 1
SB_CLASSIFICATION_THRESHOLD = 0.35
SB_N_FRAMES = 100

HYDRA_MODEL_REVISION_ATTEMPTS = 5

# Repair parameters for ScienceBirds
SB_REPAIR_TIMEOUT = 180
SB_REPAIR_MAX_ITERATIONS = 30
SB_CONSISTENCY_THRESHOLD = 50
SB_ANOMOLY_DETECTOR_THRESHOLD = 0.55

SB_REWARD_CONSISTENCY_THRESHOLD = 0.5
SB_LEVEL_NOVELTY_DETECTION_ENSEMBLE_THRESHOLD = 0.8

SCIENCE_BIRDS_SERVER_CMD = 'java -jar {}'.format(path.join(SCIENCE_BIRDS_BIN_DIR, 'linux', 'game_playing_interface.jar'))
POLYCRAFT_SERVER_CMD = "./gradlew --no-daemon --stacktrace runclient"   # Must be run in pal/PolycraftAIGym
POLYCRAFT_HEADLESS = "xvfb-run -s '-screen 0 1280x1024x24'" # Prepend to Polycraft run command to run headless
POLYCRAFT_NON_NOVELTY_LEVEL_DIR = path.join(POLYCRAFT_DIR, "pogo_100_PN")   # Path to Polycraft pre-novelty levels
POLYCRAFT_NOVELTY_LEVEL_DIR = path.join(POLYCRAFT_DIR, "shared_novelty", "POGO")
POLYCRAFT_LEVEL_DIR = path.join(ROOT_PATH, 'bin', 'pal', 'POGO_100_PN')  # Path to the polycraft levels directory.  NOTE: Please update this in your "local_settings.py" (create it if it doesn't exist)
POLYCRAFT_MAX_EXPLORATION_PLANNING_ATTEMPTS = 2 # Maximum number of times the polycraft agent will try to find a plan for an exploration task before giving up
POLYCRAFT_MAX_GENERATED_NODES = 10000  # Maximum number of nodes generated until planning is halted

SB_PLANNING_DOCKER_PATH = path.join(ROOT_PATH, 'agent', 'planning', 'sb_planning')
CARTPOLE_PLANNING_DOCKER_PATH = path.join(ROOT_PATH, 'agent', 'planning', 'cartpole_planning')
CARTPOLEPLUSPLUS_PLANNING_DOCKER_PATH = path.join(ROOT_PATH, 'agent', 'planning', 'cartpoleplusplus_planning')
POLYCRAFT_PLANNING_DOCKER_PATH = path.join(ROOT_PATH, 'agent', 'planning', 'polycraft_planning')

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


EXPERIMENT_NAME = "ENSEMBLE2_yoni_domain"
NOVELTY_TYPE = ''
