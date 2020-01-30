import sys
import os
from os import path

OS_ROOT_PATH = path.abspath(os.sep)
ROOT_PATH = path.join(path.dirname(path.dirname(path.abspath(__file__))))
if sys.platform == 'darwin':
    SCIENCE_BIRDS_BIN_DIR = path.join(ROOT_PATH,'bin')
else:
    SCIENCE_BIRDS_BIN_DIR = path.join(ROOT_PATH, 'bin')

SCIENCE_BIRDS_SERVER_CMD = 'java -jar {}'.format(path.join(ROOT_PATH,'bin','game_playing_interface.jar'))

PLANNING_DOCKER_PATH = path.join(ROOT_PATH, 'agent', 'planning', 'docker_scripts')

sc_json_config =  [{
    "host": "127.0.0.1",
    "port": "2004",
    "requestbufbytes": 4,
    "d": 4,
    "e": 5}]