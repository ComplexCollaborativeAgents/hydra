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
import demo.naive_agent_groundtruth as na
from client.agent_client import GameState
from shapely.geometry import box
#WP imports
import signal
import pickle
from os import path


class SBState(State):
    """Current State of Science Birds"""
    id = 0
    def __init__(self,objs):
        super().__init__()
        self.objects = objs


    def get_rl_id(self):
        return self.id


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
        if launch:
            self.launch_SB()
            time.sleep(1)
        else:
            self.cur_state = self.load_from_serialized_state(path.join(settings.ROOT_PATH, 'data', 'science_birds', 'serialized_levels', 'level-00.p'))
        self.create_interface()

    def kill(self):
#        cmd = '{}/kill_ab.sh'.format(settings.SCIENCE_BIRDS_BIN_DIR)
#        subprocess.run(cmd,shell=True)
        os.kill(self.SB_server_process.pid+1, signal.SIGKILL)
        self.SB_server_process.kill()
        self.SB_process.kill()



    def launch_SB(self):
        """
        Maybe this would be better in a shell script than in python
        """
        print('launching science birds')
        cmd = ''
        if sys.platform=='darwin':
            cmd='open {}/ab.app'.format(settings.SCIENCE_BIRDS_BIN_DIR)
        else:
            cmd='{}/ScienceBirds_Linux/science_birds_linux.x86_64'.format(settings.SCIENCE_BIRDS_BIN_DIR)
        # Not sure if run will work this way on ubuntu...
        self.SB_process = subprocess.Popen(cmd,stdout=subprocess.PIPE, stderr=subprocess.STDOUT,shell=True)
        print('launching java interface')
        # Popen is necessary as we have to run it in the background
        self.SB_server_process = subprocess.Popen(settings.SCIENCE_BIRDS_SERVER_CMD,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,shell=True)
        # print(self.SB_server_process.communicate()[0])
        print('done')

    def create_interface(self):
        self.sb_client = na.ClientNaiveAgent()
        self.init_first_level()


    def init_first_level(self):
        self.sb_client.ar.configure(self.sb_client.id)
        levels = self.sb_client.update_no_of_levels()

        self.sb_client.solved = [0 for x in range(levels)]
        self.sb_client.current_level = self.sb_client.get_next_level()
        self.sb_client.ar.load_level(self.sb_client.current_level)
        print('solving level: {}'.format(self.sb_client.current_level))

    def available_actions(self,state=None):
        """
        There is only one parameterized action in angry birds. Are we going to
        discretize it?
        """
        assert None

    def act(self,action):
        '''returns the new current state and reward'''
        self.history.append(action)
        ref_point = self.sb_client.tp.get_reference_point(self.cur_sling)
        dx = int(action.x - ref_point.X)
        dy = int(action.y - ref_point.Y)
        # this blocks until scene is doing
        self.sb_client.ar.shoot(ref_point.X, ref_point.Y, dx, dy, 0, action.tap, False)
        reward = 0 # this will require getting the score before and after
        self.get_current_state()
        return self.cur_state, reward, self.sb_client.ar.get_game_state() is not GameState.PLAYING

    def get_current_state(self):
        """
        side effects to set the current game status and sling objects on the environment
        """
        ground_truth_type = 'groundTruth'

        vision = self.sb_client._updateReader(ground_truth_type)

        if isinstance(vision, GameState):
            return vision

        if self.sb_client.showGroundTruth:
            vision.showResult()

        sling = vision.find_slingshot_mbr()[0]
        # TODO: look into the width and height issue of Traj planner
        sling.width, sling.height = sling.height, sling.width
        objs = {}
        for key,value in vision.allObj.items():
            for obj in value:
                objs[obj.id] = {'type' : key ,
                                'bbox': box(min(obj.points[1]),
                                            min(obj.points[0]),
                                            max(obj.points[1]),
                                            max(obj.points[0]))}
        self.cur_sling = sling
        self.cur_game_window = self.sb_client.ar.get_game_state()
        self.cur_state = SBState(objs)
        return self.cur_state

    def serialize_current_state(self, level_filename):
        pickle.dump(self.cur_state, open(level_filename, 'wb'))

    def load_from_serialized_state(self, level_filename):
        loaded_state = pickle.load(open(level_filename, 'rb'))
        return loaded_state