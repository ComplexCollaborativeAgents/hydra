import sys
import os
import settings
print(sys.path)
sys.path.append(os.path.join(settings.ROOT_PATH, 'worlds', 'science_birds_interface'))
print(sys.path)
from utils.state import State, Action, World
import trajectory_planner.trajectory_planner as tp
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
        self.sling = None


    def get_rl_id(self):
        return self.id

    def serialize_current_state(self, level_filename):
        pickle.dump(self, open(level_filename, 'wb'))

    def load_from_serialized_state(level_filename):
        return pickle.load(open(level_filename, 'rb'))

    def translate_state_to_pddl(self):

        bird_params = ''
        pig_params = ''
        block_params = ''
        goal_conds = ''

        # vision = self.sb_client._updateReader('groundTruth')

        prob_instance = '(define (problem angry_birds_prob)\n'
        prob_instance += '(:domain angry_birds_scaled)\n'
        prob_instance += '(:objects '

        birds = []
        pigs = []
        blocks = []
        platforms = []
        slingshot = {}

        for o in self.objects.items():
            if o[1]['type'] == 'pig':
                pigs.append(o)
            elif 'Bird' in o[1]['type']:
                birds.append(o)
            elif o[1]['type'] == 'wood':
                blocks.append(o)
            elif o[1]['type'] == 'hill':
                platforms.append(o)
            elif o[1]['type'] == 'slingshot':
                slingshot = o

        print('\nSLINGSHOT: ' + str(slingshot))
        print('BIRDS: ' + str(birds))
        print('PIGS: ' + str(pigs))
        print('BLOCKS: ' + str(blocks))
        print('PLATFORMS: ' + str(platforms) + '\n')


        groundOffset = slingshot[1]['bbox'].bounds[3]

        # bbox.bounds = (minX, minY, maxX, maxY);
        for bo in birds:
            prob_instance += '{}_{} '.format(bo[1]['type'], bo[0])
            bird_params += '    (not (bird_dead {}_{}))\n'.format(bo[1]['type'], bo[0])
            bird_params += '    (not (bird_released {}_{}))\n'.format(bo[1]['type'], bo[0])
            bird_params += '    (= (x_bird {}_{}) {})\n'.format(bo[1]['type'], bo[0],  round((bo[1]['bbox'].bounds[0] + bo[1]['bbox'].bounds[2])/2)-0)
            bird_params += '    (= (y_bird {}_{}) {})\n'.format(bo[1]['type'], bo[0], round(abs(((bo[1]['bbox'].bounds[1] + bo[1]['bbox'].bounds[3])/2) - groundOffset)-0))
            bird_params += '    (= (v_bird {}_{}) 270)\n'.format(bo[1]['type'], bo[0])
            bird_params += '    (= (vy_bird {}_{}) 0)\n'.format(bo[1]['type'], bo[0])
            goal_conds += ' (not (bird_dead {}_{}))'.format(bo[1]['type'], bo[0])

        prob_instance += '- bird '

        for po in pigs:
            prob_instance += '{}_{} '.format(po[1]['type'], po[0])
            pig_params += '    (not (pig_dead {}_{}))\n'.format(po[1]['type'], po[0])
            pig_params += '    (= (x_pig {}_{}) {})\n'.format(po[1]['type'], po[0], po[1]['bbox'].bounds[0])
            pig_params += '    (= (y_pig {}_{}) {})\n'.format(po[1]['type'], po[0], abs(po[1]['bbox'].bounds[1] - groundOffset))
            pig_params += '    (= (margin_pig {}_{}) {})\n'.format(po[1]['type'], po[0], round(abs(po[1]['bbox'].bounds[2] - po[1]['bbox'].bounds[0])*0.5))
            goal_conds += ' (pig_dead {}_{})'.format(po[1]['type'], po[0])

        prob_instance += '- pig '

        if blocks != []:
            for blo in blocks:
                prob_instance += '{}_{} - block '.format(blo[1]['type'], blo[0])
                bird_params += '    (not (block_destroyed {}_{}))\n'.format(blo[1]['type'], blo[0])
                bird_params += '    (= (x_block {}_{}) {})\n'.format(blo[1]['type'], blo[0],  blo[1]['bbox'].bounds[0])
                bird_params += '    (= (y_block {}_{}) {})\n'.format(blo[1]['type'], blo[0], abs(blo[1]['bbox'].bounds[1] - groundOffset))
                bird_params += '    (= (block_height {}_{}) {})\n'.format(blo[1]['type'], blo[0], abs(blo[1]['bbox'].bounds[3] - blo[1]['bbox'].bounds[1]))
                bird_params += '    (= (block_width {}_{}) {})\n'.format(blo[1]['type'], blo[0], abs(blo[1]['bbox'].bounds[2] - blo[1]['bbox'].bounds[0]))
        else:
            prob_instance += 'dummy_block - block '

        if platforms != []:
            for pla in platforms:
                prob_instance += '{}_{} - platform '.format(pla[1]['type'], pla[0])
                bird_params += '    (= (x_platform {}_{}) {})\n'.format(pla[1]['type'], pla[0],  pla[1]['bbox'].bounds[0])
                bird_params += '    (= (y_platform {}_{}) {})\n'.format(pla[1]['type'], pla[0], abs(pla[1]['bbox'].bounds[1] - groundOffset))
                bird_params += '    (= (platform_height {}_{}) {})\n'.format(pla[1]['type'], pla[0], abs(pla[1]['bbox'].bounds[3] - pla[1]['bbox'].bounds[1]))
                bird_params += '    (= (platform_width {}_{}) {})\n'.format(pla[1]['type'], pla[0], abs(pla[1]['bbox'].bounds[2] - pla[1]['bbox'].bounds[0]))
        else:
            prob_instance += 'dummy_platform - platform '

        prob_instance += ')\n' #close objects

        init_params = '(:init '
        init_params += '(= (gravity) 134.2)\n    (= (angle) 0)\n    (= (angle_rate) 10)\n    (bird_in_slingshot)\n    (not (angle_adjusted))\n'

        init_params += bird_params
        init_params += pig_params

        init_params += ')\n' # close init

        prob_instance += init_params

        prob_instance += '(:goal (and {}))\n'.format(goal_conds)

        prob_instance += '(:metric minimize(total-time))\n'
        prob_instance += ')\n' # close define
        # print(prob_instance)

        return prob_instance


class SBAction(Action):
    """first a bird, x,y position of first tap,and then the time of the second tap"""
    def __init__(self,x,y,tap,ref_x,ref_y):
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


    def __init__(self,sel_level=0,launch=True):
        self.id = 2228
        self.tp = tp.SimpleTrajectoryPlanner()
        if launch:
            self.launch_SB()
            time.sleep(5)
        self.create_interface(sel_level)


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


    def create_interface(self,first_level=0):
        with open('worlds/science_birds_interface/client/server_client_config.json', 'r') as config:
            sc_json_config = json.load(config)
        self.sb_client = ac.AgentClient(sc_json_config[0]['host'], sc_json_config[0]['port'])
        self.sb_client.connect_to_server()
        self.sb_client.configure(self.id)
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

    def act(self,action):
        '''returns the new current state and reward'''
        self.history.append(action)
        prev_score = self.sb_client.get_current_score()
        # this blocks until scene is doing
        ret = self.sb_client.shoot(action.ref_x, action.ref_y, action.dx, action.dy, 0, action.tap, False)
        if ret == 1:
            assert False
        reward =  self.sb_client.get_current_score() - prev_score
        self.get_current_state()
        return self.cur_state, reward, self.cur_game_window is not ac.GameState.PLAYING

    def get_current_state(self):
        """
        side effects to set the current game status and sling objects on the environment
        """
        image, ground_truth = self.sb_client.get_ground_truth_with_screenshot()
        self.cur_game_window = self.sb_client.get_game_state()
        self.cur_state = SBState(ground_truth,image)
        return self.cur_state

