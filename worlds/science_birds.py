import json
import os
import pickle
import subprocess
import sys
import threading
import time
from os import path

import settings
import worlds.science_birds_interface.client.agent_client as ac
import worlds.science_birds_interface.trajectory_planner.trajectory_planner as tp
from utils.state import State, Action, World
#import shapely.geometry as geo

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

    def load_from_serialized_state(level_filename):
        return pickle.load(open(level_filename, 'rb'))

    def fluents_for_objects(self):
        '''
        This returns a dictionary of fluent and values of the form produced in plan_trace
        This should be converted to the pypddl representation
        Hopefully this intermediate step won't be double the work
        '''
        # There is an annoying disconnect in representations.
        # 'x_pig[pig_4]:450' vs. (= (x_pig pig4) 450)
        # 'pig_dead[pig_4]:False vs. (not (pig_dead pig_4))
        # we need to compute an intermediate representation
        # [pred, arg, value]
        objects  = [] #name, type
        facts = []
        goals = []
        groundOffset = self.sling.bounds[3]
        bird_index = 0
        for o in self.objects.items():
            if o[1]['type'] == 'pig':
                obj_name = '{}_{}'.format(o[1]['type'], o[0])
                facts.append(['pig_dead',obj_name, False])
                facts.append(['x_pig',obj_name,o[1]['bbox'].bounds[0]])
                facts.append(['y_pig',obj_name,abs(o[1]['bbox'].bounds[1] - groundOffset)])
                facts.append(['margin_pig', obj_name, round(abs(o[1]['bbox'].bounds[2] - o[1]['bbox'].bounds[0]) * 0.75)])
                goals.append(['pig_dead', obj_name, True])
                objects.append([obj_name,o[1]['type'], o[0]])
            elif 'Bird' in o[1]['type']:
                #looks like this assumes all birds are at the sling
                obj_name = '{}_{}'.format(o[1]['type'], o[0])
                objects.append([obj_name,o[1]['type']])
                facts.append(['bird_dead',obj_name,False])
                facts.append(['bird_released',obj_name,False])
                facts.append(['x_bird',obj_name,round((slingshot[1]['bbox'].bounds[0] + self.sling['bbox'].bounds[2]) / 2) - 0])
                facts.append(['y_bird',obj_name,round(abs(((self.sling['bbox'].bounds[1] + self.sling['bbox'].bounds[3]) / 2) - groundOffset) - 0)])
                facts.append(['v_bird',obj_name, 270])
                facts.append(['vy_bird',obj_name, 0])
                facts.append(['bird_id',obj_name,bird_index])
                bird_index += 1
            elif o[1]['type'] == 'wood' or o[1]['type'] == 'ice' or o[1]['type'] == 'stone':
                blocks.append(o)
            elif o[1]['type'] == 'hill':
                platforms.append(o)
            elif o[1]['type'] == 'slingshot':
                slingshot = o

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
            elif o[1]['type'] == 'wood' or o[1]['type'] == 'ice' or o[1]['type'] == 'stone':
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

        if blocks != []:
            for bl in blocks:
                prob_instance += '{}_{} '.format(bl[1]['type'], bl[0])
                block_params += '    (= (x_block {}_{}) {})\n'.format(bl[1]['type'], bl[0], bl[1]['bbox'].bounds[0])
                block_params += '    (= (y_block {}_{}) {})\n'.format(bl[1]['type'], bl[0],
                                                                     abs(bl[1]['bbox'].bounds[1] - groundOffset))
                block_params += '    (= (block_height {}_{}) {})\n'.format(bl[1]['type'], bl[0], abs(
                    bl[1]['bbox'].bounds[3] - bl[1]['bbox'].bounds[1]))
                block_params += '    (= (block_width {}_{}) {})\n'.format(bl[1]['type'], bl[0], abs(
                    bl[1]['bbox'].bounds[2] - bl[1]['bbox'].bounds[0]))

                block_life_multiplier = 1.0
                block_mass_coeff = 1.0
                if bl[1]['type'] == 'wood':
                    block_life_multiplier = 1.0
                    block_mass_coeff = 0.375
                elif bl[1]['type'] == 'ice':
                    block_life_multiplier = 0.5
                    block_mass_coeff = 0.125
                elif bl[1]['type'] == 'stone':
                    block_life_multiplier = 2.0
                    block_mass_coeff = 1.2
                else:
                    block_life_multiplier = 2.0
                    block_mass_coeff = 1.2

                block_params += '    (= (block_life {}_{}) {})\n'.format(bl[1]['type'], bl[0], str(265 * block_life_multiplier))
                block_params += '    (= (block_mass {}_{}) {})\n'.format(bl[1]['type'], bl[0], str(block_mass_coeff))

                prob_instance += '- block '
        else:
            prob_instance += 'dummy_block - block '

        if platforms != []:
            for pla in platforms:
                prob_instance += '{}_{} - platform '.format(pla[1]['type'], pla[0])
                block_params += '    (= (x_platform {}_{}) {})\n'.format(pla[1]['type'], pla[0],
                                                                        pla[1]['bbox'].bounds[0])
                block_params += '    (= (y_platform {}_{}) {})\n'.format(pla[1]['type'], pla[0],
                                                                        abs(pla[1]['bbox'].bounds[1] - groundOffset))
                block_params += '    (= (platform_height {}_{}) {})\n'.format(pla[1]['type'], pla[0], abs(
                    pla[1]['bbox'].bounds[3] - pla[1]['bbox'].bounds[1]))
                block_params += '    (= (platform_width {}_{}) {})\n'.format(pla[1]['type'], pla[0], abs(
                    pla[1]['bbox'].bounds[2] - pla[1]['bbox'].bounds[0]))
        else:
            prob_instance += 'dummy_platform - platform '

        prob_instance += ')\n'  # close objects

        init_params = '(:init '
        init_params += '(= (gravity) 134.2)\n    (= (active_bird) 0)\n    (= (angle) 0)\n    (= (angle_rate) 20)\n    (not (angle_adjusted))\n'

        init_params += bird_params
        init_params += pig_params
        init_params += block_params

        init_params += ')\n' # close init

        prob_instance += init_params

        prob_instance += '(:goal (and {}))\n'.format(goal_conds)

        prob_instance += '(:metric minimize(total-time))\n'
        prob_instance += ')\n' # close define
        # print(prob_instance)

        return prob_instance


class SBAction(Action):
    '''Science Birds Action'''

class SBLoadLevel(SBAction):
    '''Loads the specific level'''
    def __init__(self,level):
        self.level = level


class SBShoot(SBAction):
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
    intermediate_states = []
    lock = threading.Lock()

    def __init__(self,sel_level=0,launch=True):
        self.id = 2228
        self.tp = tp.SimpleTrajectoryPlanner()
        if launch:
            self.launch_SB()
            time.sleep(1)
        self.create_interface(sel_level)



    def kill(self):
        print("Killing process groups: {}, {}".format(self.SB_server_process.pid,
                                                     self.SB_process.pid))
        try:
            os.killpg(self.SB_process.pid,9)
            os.killpg(self.SB_server_process.pid,9)
            self.gt_thread.kill()

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
            cmd='{}/sciencebirds_linux/sciencebirds_linux.x86_64 {}'. \
                format(settings.SCIENCE_BIRDS_BIN_DIR,
                       '-batchmode -nographics' if settings.HEADLESS else '')
        # Not sure if run will work this way on ubuntu...
        self.SB_process = subprocess.Popen(cmd,stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT,
                                           shell=True,
                                           start_new_session=True)
        print('launching science birds interface:{}'.format(str(self.SB_process.pid)))
        time.sleep(4)
        # Popen is necessary as we have to run it in the background
        cmd2 = '{}{}'.format('xvfb-run ' if settings.HEADLESS else '',
                             settings.SCIENCE_BIRDS_SERVER_CMD)
        self.SB_server_process = subprocess.Popen(cmd2,
                                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT,shell=True,
                                                  start_new_session=True
                                                  )
        print('launching java birds : {}'.format(str(self.SB_server_process.pid)))
        print('done')





    def create_interface(self,first_level=None):
        with open(str(path.join(settings.ROOT_PATH, 'worlds', 'science_birds_interface', 'client', 'server_client_config.json')), 'r') as config:
            sc_json_config = json.load(config)
        self.sb_client = ac.AgentClient(sc_json_config[0]['host'], sc_json_config[0]['port'])
        self.sb_client.connect_to_server()
        self.sb_client.configure(self.id)
        if first_level:
            self.init_selected_level(first_level)
        with open(str(path.join(settings.ROOT_PATH, 'worlds', 'science_birds_interface', 'client', 'server_observer_client_config.json')), 'r') as observer_config:
            observer_sc_json_config = json.load(observer_config)
        self.sb_observer = ac.AgentClient(observer_sc_json_config[0]['host'],observer_sc_json_config[0]['port'])
        self.sb_observer.connect_to_server()
        self.sb_observer.configure(self.id)

    def init_selected_level(self, s_level):
        self.current_level = s_level
        self.sb_client.load_level(self.current_level)



    def available_actions(self,state=None):
        """
        There is only one parameterized action in angry birds. Are we going to
        discretize it?
        """
        assert None

    def sample_state(self, frequency=0.5):
        """
         sample a state from the observer agent
         this method allows to be run in a different thread
         NOTE: Setting the frequency too high, i.e. <0.01 may cause lag in science birds game
               due to the calculation of the groundtruth
        """
        count = 0
        self.intermediate_states = []
        while True:
#            print('sampling {}'.format(count))
            count+=1
            ground_truth = self.sb_observer.get_ground_truth_without_screenshot()
            state = SBState(ground_truth, None, ac.GameState.UNKNOWN)
            self.intermediate_states.append(state)
#            print('sampling sleep')
            time.sleep(frequency)
            if self.lock.acquire(False):
#                print('thread exiting')
                break
        self.lock.release()
#        print('ending sampling')


    def act(self,action):
        '''returns the new current state and reward'''
        if isinstance(action,SBShoot):
            self.history.append(action)
            prev_score = self.sb_client.get_current_score()
            # this blocks until scene is doing
            self.lock.acquire()
            self.gt_thread = threading.Thread(target=self.sample_state)
            self.gt_thread.start()
            ret = self.sb_client.shoot(action.ref_x, action.ref_y, action.dx, action.dy, 0, action.tap, False)
            self.lock.release()
            self.gt_thread.join()
            if ret == 0:
                assert False
            reward =  self.sb_client.get_current_score() - prev_score
            self.get_current_state()
            return self.cur_state, reward
        elif isinstance(action,SBLoadLevel):
            self.init_selected_level(action.level)
            self.get_current_state()
            return self.cur_state, 0
        else:
            assert False

    def get_current_state(self):
        """
        side effects to set the current game status and sling objects on the environment
        """
        image = None
        if settings.SCREENSHOT:
            image, ground_truth = self.sb_client.get_ground_truth_with_screenshot()
        else:
            ground_truth = self.sb_client.get_ground_truth_without_screenshot()
        self.cur_game_window = self.sb_client.get_game_state()
        self.cur_state = SBState(ground_truth,image,self.cur_game_window)
        return self.cur_state

    def get_all_scores(self):
        return self.sb_client.get_all_level_scores()

