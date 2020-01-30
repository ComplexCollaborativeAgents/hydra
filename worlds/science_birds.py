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
import math


class SBState(State):
    """Current State of Science Birds"""
    id = 0
    def __init__(self,objs):
        super().__init__()
        self.objects = objs


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

    def translate_state_to_pddl(self):

        bird_params = ''
        pig_params = ''
        block_params = ''
        goal_conds = ''
        groundOffset = self.cur_sling.bottom_right[1]

        vision = self.sb_client._updateReader('groundTruth')

        prob_instance = '(define (problem angry_birds_prob)\n'
        prob_instance += '(:domain angry_birds_scaled)\n'
        prob_instance += '(:objects '

        for bird, bird_objs in vision.find_birds().items():
            for bo in bird_objs:
                prob_instance += '{}_{} '.format(bird, bo.id)
                bird_params += '    (not (bird_dead {}_{}))\n'.format(bird,bo.id)
                bird_params += '    (not (bird_released {}_{}))\n'.format(bird, bo.id)
                bird_params += '    (= (x_bird {}_{}) {})\n'.format(bird, bo.id,  round((min(bo.points[1]) + max(bo.points[1]))/2)-15 )
                bird_params += '    (= (y_bird {}_{}) {})\n'.format(bird, bo.id, round(abs(((min(bo.points[0]) + max(bo.points[0]))/2) - groundOffset)-15))
                bird_params += '    (= (v_bird {}_{}) 280)\n'.format(bird, bo.id)
                bird_params += '    (= (vy_bird {}_{}) 0)\n'.format(bird, bo.id)
                goal_conds += ' (not (bird_dead {}_{}))'.format(bird, bo.id)

        prob_instance += '- bird '

        for po in vision.find_pigs_mbr():
            prob_instance += '{}_{} '.format('pig', po.id)
            pig_params += '    (not (pig_dead {}_{}))\n'.format('pig', po.id)
            pig_params += '    (= (x_pig {}_{}) {})\n'.format('pig', po.id, min(po.points[1]))
            pig_params += '    (= (y_pig {}_{}) {})\n'.format('pig', po.id, abs(min(po.points[0]) - groundOffset))
            pig_params += '    (= (margin_pig {}_{}) {})\n'.format('pig', po.id, round(abs(max(po.points[1]) - min(po.points[1]))*0.5))
            goal_conds += ' (pig_dead {}_{})'.format('pig', po.id)

        prob_instance += '- pig '

        if vision.find_blocks() != None:
            for block, block_objs in vision.find_blocks().items():
                for blo in block_objs:
                    prob_instance += '{}_{} - block '.format(block, blo.id)
                    bird_params += '    (not (block_destroyed {}_{}))\n'.format(block,blo.id)
                    bird_params += '    (= (x_block {}_{}) {})\n'.format(block, blo.id,  min(blo.points[1]))
                    bird_params += '    (= (y_block {}_{}) {})\n'.format(block, blo.id, abs(min(blo.points[0]) - groundOffset))
                    bird_params += '    (= (block_height {}_{}) {})\n'.format(block, blo.id, abs(max(blo.points[0]) - min(blo.points[0])))
                    bird_params += '    (= (block_width {}_{}) {})\n'.format(block, blo.id, abs(max(blo.points[1]) - min(blo.points[1])))
        else:
            prob_instance += 'dummy_block - block '

        if vision.find_hill_mbr() != None:
            for pla in vision.find_hill_mbr():
                prob_instance += '{}_{} - platform '.format('hill', pla.id)
                bird_params += '    (= (x_platform {}_{}) {})\n'.format('hill', pla.id,  min(pla.points[1]))
                bird_params += '    (= (y_platform {}_{}) {})\n'.format('hill', pla.id, abs(min(pla.points[0]) - groundOffset))
                bird_params += '    (= (platform_height {}_{}) {})\n'.format('hill', pla.id, abs(max(pla.points[0]) - min(pla.points[0])))
                bird_params += '    (= (platform_width {}_{}) {})\n'.format('hill', pla.id, abs(max(pla.points[1]) - min(pla.points[1])))
        else:
            prob_instance += 'dummy_platform - platform '
        # prob_instance += '- platform '

        prob_instance += ')\n' #close objects

        init_params = '(:init '
        init_params += '(= (gravity) 139.0)\n    (= (angle) 0)\n    (= (angle_rate) 10)\n    (bird_in_slingshot)\n    (not (angle_adjusted))\n'

        init_params += bird_params
        init_params += pig_params

        init_params += ')\n' # close init

        prob_instance += init_params

        prob_instance += '(:goal (and {}))\n'.format(goal_conds)

        prob_instance += '(:metric minimize(total-time))\n'
        prob_instance += ')\n' # close define
        # print(prob_instance)

        return prob_instance