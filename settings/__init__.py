import sys
import os
from os import path

# Create your own local_settings.py file in this directory if you want
# to override this variable and not run headless
HEADLESS = True

OS_ROOT_PATH = path.abspath(os.sep)
ROOT_PATH = path.join(path.dirname(path.dirname(path.abspath(__file__))))
if sys.platform == 'darwin':
    SCIENCE_BIRDS_BIN_DIR = path.join(ROOT_PATH,'bin')
else:
    SCIENCE_BIRDS_BIN_DIR = path.join(ROOT_PATH, 'bin')

SCIENCE_BIRDS_SERVER_CMD = 'java -jar {}'.format(path.join(ROOT_PATH,'bin','game_playing_interface.jar'))

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
