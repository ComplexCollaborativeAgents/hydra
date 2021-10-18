import json
import os
import pickle
import subprocess
import sys
import threading
import time
from os import path
import copy
import math

import func_timeout

import settings
import worlds.science_birds_interface.client.agent_client as ac
from worlds.science_birds_interface.trajectory_planner.trajectory_planner import SimpleTrajectoryPlanner
from utils.host import Host
from agent.planning.pddlplus_parser import PddlPlusProblem
from utils.state import State, Action, World
#import shapely.geometry as geo
import logging


logging.basicConfig(format='%(name)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Science Birds")

class SBState(State):
    """Current State of Science Birds"""
    id = 0
    def __init__(self,objs,image,game_state):
        super().__init__()
        self.objects = objs
        self.image = image
        self.game_state = game_state
        self.sling = None

    def summary(self):
        '''returns a summary of state'''
        ret = {}
        for key, obj in self.objects.items():
            ret['{}_{}'.format(obj['type'],key)] = (obj['bbox'].centroid.x,obj['bbox'].centroid.y)
        return ret

    def get_rl_id(self):
        return self.id

    def serialize_current_state(self, level_filename):
        pickle.dump(self, open(level_filename, 'wb'))

    def load_from_serialized_state(self, level_filename):
        return pickle.load(open(level_filename, 'rb'))


class SBAction(Action):
    """Science Birds Action"""

class SBLoadLevel(SBAction):
    """Loads the specific level"""
    def __init__(self, level):
        super().__init__()
        self.level = level


class SBShoot(SBAction):
    """fires a bird, x,y position of first tap,and then the time of the second tap"""
    def __init__(self, x, y, tap, ref_x, ref_y):
        super().__init__()
        self.dx =  int(x - ref_x)
        self.dy =  int(y - ref_y)
        self.tap = tap
        self.ref_x = ref_x
        self.ref_y = ref_y

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
    intermediate_states = []
    lock = threading.Lock()

    trajectory_planner = SimpleTrajectoryPlanner() # This is static to allow others to reason about it

    def __init__(self, sel_level=0, launch=False, config='test_config.xml', host=None):
        super().__init__()
        self.id = 2228
        self.SB_process = None
        self.SB_server_process = None
        if launch:
            self.launch_SB(config)
            time.sleep(5)
        self.create_interface(sel_level, host=host)



    def kill(self):
        if self.SB_server_process:
            logger.info("Killing process groups: {}".format(self.SB_server_process.pid))
            try:
                os.killpg(self.SB_server_process.pid,9)
            except:
                logger.info("Error during process terminatio6n")
                pass

    def launch_SB(self,config='test_config.xml'):
        """
        Maybe this would be better in a shell script than in python
        """
        print('launching science birds')
        cmd = ''

        # if sys.platform=='darwin':
        #     if config:
        #         cmd='open {}/ScienceBirds_MacOS.app --args --configpath {}/data/science_birds/config/{}'.format(
        #             settings.SCIENCE_BIRDS_BIN_DIR,settings.ROOT_PATH,config)
        #     else:
        #         cmd = 'open {}/ScienceBirds_MacOS.app'.format(settings.SCIENCE_BIRDS_BIN_DIR)
        # else:
        #     if config:
        #         cmd='{}/sciencebirds_linux/sciencebirds_linux.x86_64 --configpath {}/data/science_birds/config/{}'. \
        #             format(settings.SCIENCE_BIRDS_BIN_DIR, settings.ROOT_PATH,config)
        #     else:
        #         cmd='{}/sciencebirds_linux/sciencebirds_linux.x86_64'.format(settings.SCIENCE_BIRDS_BIN_DIR)
        # # Not sure if run will work this way on ubuntu...
        # self.SB_process = subprocess.Popen(cmd,stdout=subprocess.PIPE,
        #                                    stderr=subprocess.STDOUT,
        #                                    shell=True,
        #                                    start_new_session=True)
        # print('launching science birds interface:{}'.format(str(self.SB_process.pid)))
        # time.sleep(4)
        # Popen is necessary as we have to run it in the background

#        cmd = 'cp {}/data/science_birds/config/{} {}/linux/config.xml'.format(str(settings.ROOT_PATH), config, settings.SCIENCE_BIRDS_BIN_DIR)
#        subprocess.run(cmd, shell=True)

        cmd2 = 'cd {} && {} {} {} {} > game_playing_interface.log'.format(settings.SCIENCE_BIRDS_BIN_DIR + "/linux/",
                                                                       settings.SCIENCE_BIRDS_SERVER_CMD,
                                                                       '--config-path {}'.format(os.path.join(settings.ROOT_PATH,'data','science_birds','config',config)) if config else '',
                                                                       '--headless' if settings.HEADLESS else '',
                                                                       '--dev' if settings.SB_DEV_MODE else ''
                                                                       )
        print('launching java birds : {}'.format(cmd2))
        self.SB_server_process = subprocess.Popen(cmd2,
                                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT,shell=True,
                                                  start_new_session=True
                                                  )
        print('launching java birds : {}'.format(str(self.SB_server_process.pid)))
        print('done')

    def load_hosts(self, server_host: Host, observer_host: Host):
        with open(str(path.join(settings.ROOT_PATH, 'worlds', 'science_birds_interface', 'client', 'server_client_config.json')), 'r') as config:
            sc_json_config = json.load(config)

        server = Host(sc_json_config[0]['host'], sc_json_config[0]['port'])
        if 'DOCKER' in os.environ:
            server.hostname = 'docker-host'
        if server_host:
            server.hostname = server_host.hostname
            server.port = server_host.port

        with open(str(path.join(settings.ROOT_PATH, 'worlds', 'science_birds_interface', 'client', 'server_observer_client_config.json')), 'r') as observer_config:
            observer_sc_json_config = json.load(observer_config)

        observer = Host(observer_sc_json_config[0]['host'], observer_sc_json_config[0]['port'])
        if 'DOCKER' in os.environ:
            observer.hostname = 'docker-host'
        if observer_host:
            observer.hostname = observer_host.hostname
            observer.port = observer_host.port


    def create_interface(self,first_level=None,host=None):
        with open(str(path.join(settings.ROOT_PATH, 'worlds', 'science_birds_interface', 'client', 'server_client_config.json')), 'r') as config:
            sc_json_config = json.load(config)
        if host is not None:
            sc_json_config[0]['host'] = host.hostname
            sc_json_config[0]['port'] = host.port
        self.sb_client = ac.AgentClient(sc_json_config[0]['host'], sc_json_config[0]['port'])
        self.sb_client.connect_to_server()
        self.sb_client.configure(self.id)
        if first_level:
            self.init_selected_level(first_level)


    def init_selected_level(self, s_level):
        self.current_level = s_level
        self.sb_client.load_level(self.current_level)



    def available_actions(self,state=None):
        """
        There is only one parameterized action in angry birds. Are we going to
        discretize it?
        """
        assert None

    def act(self, action):
        """returns the new current state and reward"""
        if isinstance(action, SBShoot):
            logger.info("Executing action")
            self.history.append(action)
            prev_score = self.sb_client.get_current_score()

            # This blocks until the scene is static. Currently asking for every 10th frame, but it should be parameterized to sim_speed.
            if settings.SCREENSHOT and not settings.HEADLESS:
                self.intermediate_states = []

                b_img, b_gt = self.sb_client.get_ground_truth_with_screenshot()
                self.intermediate_states.append(SBState(b_gt, b_img, None))

                self.sb_client.shoot_and_record_ground_truth(action.ref_x+action.dx, action.ref_y+action.dy, 0, action.tap, settings.SB_GT_FREQ)
                time.sleep(2 / settings.SB_SIM_SPEED)
                
                a_img, a_gt = self.sb_client.get_ground_truth_with_screenshot()
                self.intermediate_states.append(SBState(a_gt, a_img, None))
            else:
                self.intermediate_states = self.sb_client.shoot_and_record_ground_truth(action.ref_x+action.dx, action.ref_y+action.dy, 0, action.tap, settings.SB_GT_FREQ)
                self.intermediate_states = [SBState(intermediate_state, None, None) for intermediate_state in self.intermediate_states]
                time.sleep(2 / settings.SB_SIM_SPEED)
    #            if len(self.intermediate_states) < 3: # we should get some intermediate states
    #                assert False
            reward =  self.sb_client.get_current_score() - prev_score
            self.get_current_state()
            logger.info("Action executed ref_pt ({},{}) action ({},{}) reward {} len(intermediate_states) {}".format(action.ref_x, action.ref_y, action.dx, action.dy,reward,len(self.intermediate_states)))
            return self.cur_state, reward
        elif isinstance(action,SBLoadLevel):
            self.init_selected_level(action.level)
            self.get_current_state()
            return self.cur_state, 0
        else:
            assert False



#    @func_timeout.func_set_timeout(2)
    def get_current_state(self):
        """
        side effects to set the current game status and sling objects on the environment
        """
        image = None
        time.sleep(0.1) #As of 0.3.7 we should not need sleeps
        self.cur_game_window = self.sb_client.get_game_state()
        if self.cur_game_window != ac.GameState.PLAYING: # if you aren't playing you can't get ground truth anymore
            return SBState(None,None,self.cur_game_window)

        if settings.SCREENSHOT and not settings.HEADLESS:
            image, ground_truth = self.sb_client.get_ground_truth_with_screenshot()
        else:
            ground_truth = self.sb_client.get_ground_truth_without_screenshot()
        self.cur_state = SBState(ground_truth,image,self.cur_game_window)
        return self.cur_state

    def get_all_scores(self):
        return self.sb_client.get_all_level_scores()


class ExternalAgent:
    def __init__(self, agent_path, agent_name):
        self.path = agent_path
        self.agent_name = agent_name

    def run(self):
        command = "cd {} && java -jar {} 1".format(self.path, self.agent_name)
        logger.info("starting agent process: {}".format(command))
        process = subprocess.Popen(command, shell=True)
        process.wait()
        logger.info("agent process ended")


class DatalabAgent(ExternalAgent):
    def __init__(self):
        super().__init__(agent_path=settings.SCIENCE_BIRDS_BIN_DIR + "/baseline_agents/",
                         agent_name='datalab.jar')


class EaglewingsAgent(ExternalAgent):
    def __init__(self):
        super().__init__(agent_path=settings.SCIENCE_BIRDS_BIN_DIR + "/baseline_agents/",
                         agent_name='ealgewings.jar')