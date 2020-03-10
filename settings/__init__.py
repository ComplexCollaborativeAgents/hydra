import sys
import os
from os import path

# Create your own local_settings.py file in this directory if you want
# to override this variable and not run headless
HEADLESS = True
SCREENSHOT = False

OS_ROOT_PATH = path.abspath(os.sep)
ROOT_PATH = path.join(path.dirname(path.dirname(path.abspath(__file__))))
SCIENCE_BIRDS_BIN_DIR = path.join(ROOT_PATH,'bin')
SCIENCE_BIRDS_LEVELS_DIR = path.join(SCIENCE_BIRDS_BIN_DIR,'sciencebirds_linux',
                                     'sciencebirds_linux_Data','StreamingAssets','Levels','novelty_level_0','type1','Levels')
SB_INIT_COLOR_MAP = path.join(ROOT_PATH,'worlds','science_birds_interface','demo','ColorMap.json')

SCIENCE_BIRDS_SERVER_CMD = 'java -jar {}'.format(path.join(SCIENCE_BIRDS_BIN_DIR,'game_playing_interface.jar'))

PLANNING_DOCKER_PATH = path.join(ROOT_PATH, 'agent', 'planning', 'docker_scripts')

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
