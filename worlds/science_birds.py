import sys
import os
import settings
print(sys.path)
sys.path.append(os.path.join(settings.ROOT_PATH, 'worlds', 'science_birds_interface'))
print(sys.path)
from utils.state import State, Action, World
import subprocess
import settings
import sys
import time
import json
import client.agent_client as ac
from shapely.geometry import box
#WP imports
import signal
import pickle
from os import path
import math


class SBState(State):
    """Current State of Science Birds"""
    id = 0
    def __init__(self,objs,image):
        super().__init__()
        self.objects = objs
        self.image = image


    def get_rl_id(self):
        return self.id

    def serialize_current_state(self, level_filename):
        pickle.dump(self, open(level_filename, 'wb'))

    def load_from_serialized_state(level_filename):
        return pickle.load(open(level_filename, 'rb'))


class SBAction(Action):
    """first a bird, x,y position of first tap, and then the time of the second tap"""
    def __init__(self,x,y,tap):
        self.x = x
        self.y = y
        self.tap = tap

    def get_rl_id(self):
        return self.to_id


class ScienceBirds(World):
    """
    Science birds interface supplied from ANU
    There is one ScienceBirds world per session
    We will make calls through the JAR to change the level
    """

    sb_client = None
    history = []

    def __init__(self,launch=True):
        self.id = 2228
        if launch:
            self.launch_SB()
            time.sleep(5)
        self.create_interface()

    def kill(self):
        print("Killing processes: {}, {}, {}".format(self.SB_server_process.pid+1,
                                                     self.SB_server_process.pid,
                                                     self.SB_process.pid))
        try:
            os.kill(self.SB_server_process.pid + 1,signal.SIGTERM )
            self.SB_process.kill()
            self.SB_server_process.kill()
        except:
            pass

            

            
    def launch_SB(self):
        """
        Maybe this would be better in a shell script than in python
        """
        print('launching science birds')
        cmd = ''

        if sys.platform=='darwin':
            cmd='open {}/ScienceBirds_MacOS.app'.format(settings.SCIENCE_BIRDS_BIN_DIR)
        else:
            cmd='{}/ScienceBirds_Linux/science_birds_linux.x86_64 {}'. \
                format(settings.SCIENCE_BIRDS_BIN_DIR,
                       '-batchmode -nographics' if settings.HEADLESS else '')
        # Not sure if run will work this way on ubuntu...
        self.SB_process = subprocess.Popen(cmd,stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT,
                                           shell=True,
                                           preexec_fn=os.setsid)
        print('launching java interface')
        # Popen is necessary as we have to run it in the background
        cmd2 = '{}{}'.format('xvfb-run ' if settings.HEADLESS else '',
                             settings.SCIENCE_BIRDS_SERVER_CMD)
        self.SB_server_process = subprocess.Popen(cmd2,
                                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT,shell=True,
                                                  preexec_fn=os.setsid)
    #        print(self.SB_server_process.communicate()[0])
        print('done')

    def create_interface(self):
        with open('worlds/science_birds_interface/client/server_client_config.json', 'r') as config:
            sc_json_config = json.load(config)
        self.sb_client = ac.AgentClient(**sc_json_config[0])
        self.sb_client.connect_to_server()
        self.init_first_level()


    def init_first_level(self):
        self.sb_client.configure(self.id)
        levels = self.sb_client.get_number_of_levels()

        self.solved = [0 for x in range(levels)]
        self.current_level = self.sb_client.load_level(1)
        self.sb_client.load_level(self.current_level)
        print('solving level: {}'.format(self.current_level))

    def available_actions(self,state=None):
        """
        There is only one parameterized action in angry birds. Are we going to
        discretize it?
        """
        assert None

    def act(self,action):
        '''returns the new current state and reward'''
        self.history.append(action)
        prev_score = self.sb_client.get_current_score()
        ref_point = self.sb_client.tp.get_reference_point(self.cur_sling)
        dx = int(action.x - ref_point.X)
        dy = int(action.y - ref_point.Y)
        # this blocks until scene is doing
        self.sb_client.ar.shoot(ref_point.X, ref_point.Y, dx, dy, 0, action.tap, False)
        reward =  self.sb_client.ab.get_current_score() - prev_score
        self.get_current_state()
        return self.cur_state, reward, self.sb_client.get_game_state() is not ac.GameState.PLAYING

    def get_current_state(self):
        """
        side effects to set the current game status and sling objects on the environment
        """
        image, ground_truth = self.sb_client.get_ground_truth_with_screenshot()
        self.cur_game_window = self.sb_client.get_game_state()
        self.cur_state = SBState(ground_truth,image)
        return self.cur_state
