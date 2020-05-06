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

import settings
import worlds.science_birds_interface.client.agent_client as ac
import worlds.science_birds_interface.trajectory_planner.trajectory_planner as tp
from agent.planning.pddl_plus import PddlPlusProblem, PddlPlusState
from utils.state import State, Action, World

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


    ''' Translate the initial state, as observed, to a PddlPlusProblem object. 
    Note that in the initial state, we ignore the location of the bird and assume it is on the slingshot. '''
    def translate_initial_state_to_pddl_problem(self):
        # There is an annoying disconnect in representations.
        # 'x_pig[pig_4]:450' vs. (= (x_pig pig4) 450)
        # 'pig_dead[pig_4]:False vs. (not (pig_dead pig_4))
        # We will use the PddlPlusProblem class as a common representations
        # init rep [['=', ['gravity'], '134.2'], ['=', ['active_bird'], '0'], ['=', ['angle'], '0'], ['=', ['angle_rate'], '20'], ['not', ['angle_adjusted']], ['not', ['bird_dead', 'redBird_0']], ['not', ['bird_released', 'redBird_0']], ['=', ['x_bird', 'redBird_0'], '192'], ['=', ['y_bird', 'redBird_0'], '29'], ['=', ['v_bird', 'redBird_0'], '270'], ['=', ['vy_bird', 'redBird_0'], '0'], ['=', ['bird_id', 'redBird_0'], '0'], ['not', ['wood_destroyed', 'wood_2']], ['=', ['x_wood', 'wood_2'], '445.0'], ['=', ['y_wood', 'wood_2'], '25.0'], ['=', ['wood_height', 'wood_2'], '12.0'], ['=', ['wood_width', 'wood_2'], '24.0'], ['not', ['wood_destroyed', 'wood_3']], ['=', ['x_wood', 'wood_3'], '447.0'], ['=', ['y_wood', 'wood_3'], '13.0'], ['=', ['wood_height', 'wood_3'], '13.0'], ['=', ['wood_width', 'wood_3'], '24.0'], ['not', ['pig_dead', 'pig_4']], ['=', ['x_pig', 'pig_4'], '449.0'], ['=', ['y_pig', 'pig_4'], '53.0'], ['=', ['margin_pig', 'pig_4'], '21']]
        # objects rep [('redBird_0', 'bird'), ('pig_4', 'pig'), ('wood_2', 'wood_block'), ('wood_3', 'wood_block'), ('dummy_ice', 'ice_block'), ('dummy_stone', 'stone_block'), ('dummy_platform', 'platform')]
        prob = PddlPlusProblem()
        prob.domain = 'angry_birds_scaled'
        prob.name = 'angry_birds_prob'
        prob.metric = 'minimize(total-time)'
        prob.objects = []
        prob.init = []
        prob.goal = []

        #we should probably use the self.sling on the object
        slingshot = None
        for o in self.objects.items():
            if o[1]['type'] == 'slingshot':
                slingshot = o

        groundOffset = slingshot[1]['bbox'].bounds[3]
        bird_index = 0

        platform = False
        block = False
        for o in self.objects.items():
            if o[1]['type'] == 'pig':
                obj_name = '{}_{}'.format(o[1]['type'], o[0])
                prob.init.append(['not', ['pig_dead',obj_name]])
                prob.init.append(['=',['x_pig',obj_name],round(abs(o[1]['bbox'].bounds[2] + o[1]['bbox'].bounds[0])/2)])
                prob.init.append(['=',['y_pig',obj_name],abs(round(abs(o[1]['bbox'].bounds[1] + o[1]['bbox'].bounds[3])/2) - groundOffset)])
                prob.init.append(['=',['pig_radius', obj_name], round((abs(o[1]['bbox'].bounds[2] - o[1]['bbox'].bounds[0])/2) * 0.75)])
                prob.init.append(['=', ['m_pig', obj_name], 1])
                prob.goal.append(['pig_dead', obj_name])
                prob.objects.append((obj_name,o[1]['type']))
            elif 'Bird' in o[1]['type']:
                obj_name = '{}_{}'.format(o[1]['type'], o[0])
                prob.objects.append((obj_name,'Bird')) #This probably needs to change
                # prob.init.append(['not',['bird_dead',obj_name]])
                prob.init.append(['not',['bird_released',obj_name]])
                prob.init.append(['=',['x_bird',obj_name],round((slingshot[1]['bbox'].bounds[0] + slingshot[1]['bbox'].bounds[2]) / 2) - 0])
                prob.init.append(['=',['y_bird',obj_name],round(abs(((slingshot[1]['bbox'].bounds[1] + slingshot[1]['bbox'].bounds[3]) / 2) - groundOffset) - 0)])
                prob.init.append(['=',['v_bird',obj_name], 270])
                prob.init.append(['=',['vx_bird',obj_name], 0])
                prob.init.append(['=',['vy_bird',obj_name], 0])
                prob.init.append(['=',['m_bird',obj_name], 1])
                prob.init.append(['=',['bounce_count',obj_name], 0])
                prob.init.append(['=',['bird_id',obj_name],bird_index])
                bird_index += 1
            elif o[1]['type'] == 'wood' or o[1]['type'] == 'ice' or o[1]['type'] == 'stone':
                block = True
                obj_name = '{}_{} '.format(o[1]['type'], o[0])

                bl_x = round((o[1]['bbox'].bounds[2] + o[1]['bbox'].bounds[0])/2)
                bl_y = abs(round(abs(o[1]['bbox'].bounds[1] + o[1]['bbox'].bounds[3])/2) - groundOffset)

                prob.init.append(['=',['x_block', obj_name], bl_x ])
                prob.init.append(['=',['y_block',obj_name], bl_y ])
                # prob.init.append(['block_supporting',obj_name])

                bl_height = abs(o[1]['bbox'].bounds[3] - o[1]['bbox'].bounds[1])
                bl_width = abs(o[1]['bbox'].bounds[2] - o[1]['bbox'].bounds[0])

                prob.init.append(['=',['block_height',obj_name], bl_height])
                prob.init.append(['=',['block_width',obj_name], bl_width])
                block_life_multiplier = 1.0
                block_mass_coeff = 1.0
                if o[1]['type'] == 'wood':
                    block_life_multiplier = 1.0
                    block_mass_coeff = 0.375*1.3
                elif o[1]['type'] == 'ice':
                    block_life_multiplier = 0.5
                    block_mass_coeff = 0.125*2
                elif o[1]['type'] == 'stone':
                    block_life_multiplier = 2.0
                    block_mass_coeff = 1.2
                elif o[1]['type'] == 'TNT':
                    block_life_multiplier = 0.001
                    block_mass_coeff = 1.2
                    prob.init.append(['block_explosive', obj_name])
                else: # not sure how this could ever happen
                    block_life_multiplier = 2.0
                    block_mass_coeff = 1.2
                prob.init.append(['=',['block_life',obj_name],str(math.ceil(265 * block_life_multiplier))])
                prob.init.append(['=',['block_mass',obj_name],str(block_mass_coeff)])

                bl_stability = 265*(bl_width/bl_height)*(1-(bl_y/groundOffset))*block_mass_coeff

                prob.init.append(['=',['block_stability',obj_name], bl_stability])

                prob.objects.append((obj_name,'block'))
            elif o[1]['type'] == 'hill':
                platform = True
                obj_name ='{}_{}'.format(o[1]['type'], o[0])
                prob.init.append(['=',['x_platform', obj_name], round((o[1]['bbox'].bounds[2] + o[1]['bbox'].bounds[0])/2)])
                prob.init.append(['=', ['y_platform', obj_name], abs(round(abs(o[1]['bbox'].bounds[1] + o[1]['bbox'].bounds[3])/2) - groundOffset)])
                prob.init.append(['=', ['platform_height', obj_name], abs(o[1]['bbox'].bounds[3] - o[1]['bbox'].bounds[1])])
                prob.init.append(['=', ['platform_width', obj_name], abs(o[1]['bbox'].bounds[2] - o[1]['bbox'].bounds[0])])
                prob.objects.append([obj_name,'platform'])
            elif o[1]['type'] == 'slingshot':
                slingshot = o
        for fact in [['=',['gravity'], 134.2],
                     ['=',['active_bird'], 0],
                     ['=', ['angle'], 0],
                     ['not', ['angle_adjusted']],
                     ['not', ['pig_killed']],
                     ['=',['angle_rate'], 10],
                     ['=', ['ground_damper'], 0.4]
                     ]:
            prob.init.append(fact)
        if not platform:
            prob.objects.append(['dummy_platform','platform'])
        if not block:
            prob.objects.append(['dummy_block','block'])

        prob_simplified = PddlPlusProblem()
        prob_simplified.name = copy.copy(prob.name)
        prob_simplified.domain = copy.copy(prob.domain)
        prob_simplified.objects = copy.copy(prob.objects)
        prob_simplified.init = copy.copy(prob.init)
        prob_simplified.metric = copy.copy(prob.metric)
        prob_simplified.goal = list()
        prob_simplified.goal.append(['pig_killed'])

        # print("\n\nPROB: " + str(prob.goal))
        # print("\nPROB SIMPLIFIED: " + str(prob_simplified.goal))

        return prob, prob_simplified

    ''' Translate an intermediate state to PddlPlusState. 
    Key difference between this method and translate_init_state... method is that here we consider the location of the birds'''
    def translate_intermediate_state_to_pddl_state(self):
        state_as_list = list()

        #we should probably use the self.sling on the object
        slingshot = None
        for o in self.objects.items():
            if o[1]['type'] == 'slingshot':
                slingshot = o
        groundOffset = slingshot[1]['bbox'].bounds[3]

        slingshot_x = round((slingshot[1]['bbox'].bounds[0] + slingshot[1]['bbox'].bounds[2]) / 2)
        slingshot_y = round(abs(((slingshot[1]['bbox'].bounds[1] + slingshot[1]['bbox'].bounds[3]) / 2) - groundOffset) - 0)
        bird_index = 0
        platform = False
        block = False

        for o in self.objects.items():
            if o[1]['type'] == 'pig':
                obj_name = '{}_{}'.format(o[1]['type'], o[0])
                state_as_list.append(['not', ['pig_dead',obj_name]])
                state_as_list.append(['=', ['x_pig',obj_name], self.compute_x_coordinate(o)])
                state_as_list.append(['=', ['y_pig',obj_name], self.compute_y_coordinate(groundOffset, o)])
                state_as_list.append(['=', ['pig_radius', obj_name], self.compute_radius(o)])
                state_as_list.append(['=', ['m_pig', obj_name], 1])
            elif 'bird' in o[1]['type'].lower():
                obj_name = '{}_{}'.format(o[1]['type'], o[0])
                # prob.init.append(['not',['bird_dead',obj_name]])
                # prob.init.append(['not',['bird_released',obj_name]])
                self.compute_x_coordinate(o)

                # Need to separate the case where we're before shooting the bird and after.
                # Before: the bird location is considered as the location of the slingshot,
                # afterwards, it's the location of the birds bounding box
                x_bird = self.compute_x_coordinate(o)
                if x_bird>slingshot_x:
                    state_as_list.append(['=',['x_bird',obj_name], self.compute_x_coordinate(o)])
                    state_as_list.append(['=',['y_bird',obj_name], self.compute_y_coordinate(groundOffset,o)])
                else:
                    state_as_list.append(['=', ['x_bird', obj_name], slingshot_x])
                    state_as_list.append(['=', ['y_bird', obj_name], slingshot_y])


                # prob.init.append(['=',['v_bird',obj_name], 270])  Computing velocity is more difficult
                # prob.init.append(['=',['vx_bird',obj_name], 0])
                # prob.init.append(['=',['vy_bird',obj_name], 0])
                state_as_list.append(['=',['m_bird',obj_name], 1])
                #prob.init.append(['=',['bounce_count',obj_name], 0])
                state_as_list.append(['=',['bird_id',obj_name],bird_index])
                bird_index += 1
            elif o[1]['type'] == 'wood' or o[1]['type'] == 'ice' or o[1]['type'] == 'stone':
                block = True
                obj_name = '{}_{} '.format(o[1]['type'], o[0])
                state_as_list.append(['=',['x_block', obj_name], round((o[1]['bbox'].bounds[2] + o[1]['bbox'].bounds[0])/2)])
                state_as_list.append(['=', ['y_block',obj_name], self.compute_y_coordinate(groundOffset, o)])
                state_as_list.append(['=',['block_height',obj_name],abs(
                    o[1]['bbox'].bounds[3] - o[1]['bbox'].bounds[1])])
                state_as_list.append(['=',['block_width',obj_name],abs(
                    o[1]['bbox'].bounds[2] - o[1]['bbox'].bounds[0])])
                block_life_multiplier = 1.0
                block_mass_coeff = 1.0
                if o[1]['type'] == 'wood':
                    block_life_multiplier = 1.0
                    block_mass_coeff = 0.375
                elif o[1]['type'] == 'ice':
                    block_life_multiplier = 0.5
                    block_mass_coeff = 0.125
                elif o[1]['type'] == 'stone':
                    block_life_multiplier = 2.0
                    block_mass_coeff = 1.2
                else: # not sure how this could ever happen
                    block_life_multiplier = 2.0
                    block_mass_coeff = 1.2
                state_as_list.append(['=',['block_life',obj_name],str(265 * block_life_multiplier)])
                state_as_list.append(['=',['block_mass',obj_name],str(block_mass_coeff)])
            elif o[1]['type'] == 'hill':
                platform = True
                obj_name ='{}_{}'.format(o[1]['type'], o[0])
                state_as_list.append(['=',['x_platform', obj_name], round((o[1]['bbox'].bounds[2] + o[1]['bbox'].bounds[0])/2)])
                state_as_list.append(['=', ['y_platform', obj_name], self.compute_y_coordinate(groundOffset, o)])
                state_as_list.append(['=', ['platform_height', obj_name], abs(o[1]['bbox'].bounds[3] - o[1]['bbox'].bounds[1])])
                state_as_list.append(['=', ['platform_width', obj_name], abs(o[1]['bbox'].bounds[2] - o[1]['bbox'].bounds[0])])
            elif o[1]['type'] == 'slingshot':
                slingshot = o
        for fact in [['=',['gravity'], 134.2],
                     ['=',['active_bird'], 0],
                     ['=', ['angle'], 0],
                     ['not', ['angle_adjusted']],
                     ['=',['angle_rate'], 10],
                     ['=', ['ground_damper'], 0.4]
                     ]:
            state_as_list.append(fact)
        return PddlPlusState(state_as_list)

    ''' Computes the y coordinate of the given object as the center of its bounding box, 
    corrected for the given groundOffset '''
    def compute_y_coordinate(self, groundOffset, o):
        return abs(round(abs(o[1]['bbox'].bounds[1] + o[1]['bbox'].bounds[3]) / 2) - groundOffset)

    ''' Computes the x coordinate of the given object as the center of its boundingbox.'''
    def compute_x_coordinate(self, o):
        return round(abs(o[1]['bbox'].bounds[2] + o[1]['bbox'].bounds[0]) / 2)

    ''' Computes the radius of the given object '''
    def compute_radius(self, o):
        return round((abs(o[1]['bbox'].bounds[2] - o[1]['bbox'].bounds[0]) / 2) * 0.75)


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

    def __init__(self,sel_level=0,launch=False):
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
            cmd='open {}/ScienceBirds_MacOS.app --args --configpath {}/data/science_birds/config/test_config.xml'.format(
                settings.SCIENCE_BIRDS_BIN_DIR,settings.ROOT_PATH)
        else:
            cmd='{}/sciencebirds_linux/sciencebirds_linux.x86_64 --configpath {}/data/science_birds/config/test_config.xml'. \
                format(settings.SCIENCE_BIRDS_BIN_DIR, settings.ROOT_PATH)
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

