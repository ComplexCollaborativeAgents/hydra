import math
from agent.planning.pddlplus_parser import *
import settings
import logging
from agent.planning.meta_model import *
import numpy as np

fh = logging.FileHandler("cartpole_hydra.log",mode='w')
formatter = logging.Formatter('%(asctime)-15s %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger = logging.getLogger("cartpole_pddl_meta_model")
logger.setLevel(logging.INFO)
logger.addHandler(fh)


''' A generator for Pddl Objects '''
class PddlObjectType():
    ''' Accepts an object from SBState.objects '''
    def __init__(self):
        self.hyper_parameters = dict()
        self.pddl_type="object" # This the PDDL+ type of this object.

    ''' Subclasses should override this setting all attributes of that object '''
    def _compute_obj_attributes(self, obj, problem_params:dict):
        return dict()

    ''' Subclasses should override this settign all attributes of that object that can be observed'''
    def _compute_observable_obj_attributes(self, obj, problem_params:dict):
        return dict()

    ''' Populate a PDDL+ problem with details about this object '''
    def add_object_to_problem(self, prob: PddlPlusProblem, obj, problem_params:dict):
        name = self._get_name(obj)
        prob.objects.append([name, self.pddl_type])
        attributes = self._compute_obj_attributes(obj, problem_params)
        for attribute in attributes:
            value = attributes[attribute]
            # If attribute is Boolean no need for an "=" sign
            if isinstance(value,  bool):
                if value==True:
                    prob.init.append([attribute, name])
                else: # value == False
                    prob.init.append(['not', [attribute, name]])
            else: # Attribute is a number
                prob.init.append(['=', [attribute, name], value])

    ''' Populate a PDDL+ state with details about this object '''
    def add_object_to_state(self, pddl_state: PddlPlusState, obj, state_params:dict):
        name = self._get_name(obj)
        attributes = self._compute_observable_obj_attributes(obj, state_params)
        for attribute in attributes:
            value = attributes[attribute]
            fluent_name = (attribute, name)
            # If attribute is Boolean no need for an "=" sign
            if isinstance(value,  bool):
                if value==True:
                    pddl_state.boolean_fluents.add(fluent_name)
                # TODO: Think how to handle booean fluents with False value. Not as trivial as it sounds
            else: # Attribute is a number
                pddl_state.numeric_fluents[fluent_name]=value


class CartPolePlusPlusMetaModel(MetaModel):
    PLANNER_PRECISION = 5 # how many decimal points the planner can handle correctly

    def __init__(self):
        super().__init__(
            docker_path=settings.CARTPOLEPLUSPLUS_PLANNING_DOCKER_PATH,
            domain_file_name="cartpole_plus_plus_domain.pddl",
            delta_t=settings.CP_DELTA_T,
            metric='minimize(total-time)',
            repairable_constants=('m_cart', 'l_pole', 'm_pole', 'force_mag', 'gravity', 'angle_limit', 'x_limit'),
            repair_deltas=(1.0, 0.1, 0.1, 1.0, 1.0, 0.1, 0.1),
            constant_numeric_fluents={
                'm_cart': 1.0,
                'friction_cart': 0.0,
                'l_pole': 1.0,
                # 'l_pole': 0.5,
                'm_pole': 0.1,
                'friction_pole': 0.0,
                'force_mag': 10.0,
                'inertia': 1.0,
                'elapsed_time': 0.0,
                'gravity': 9.81,
                'time_limit': 1.0,
                'angle_limit': 0.165, # ~10 degrees in rad
                # 'angle_limit': 0.205, # ~12 degrees in rad
                # 'pos_limit': 2.4,
                'wall_x_min': -5,
                'wall_x_max': 5,
                'wall_y_min': -5,
                'wall_y_max': 5,
                'wall_z_min': 0,
                'wall_z_max': 10,
                },
            constant_boolean_fluents={
                'total_failure':False,
                'ready':True,
                'cart_available':True})

    ''' Translate the initial SBState, as observed, to a PddlPlusProblem object. 
    Note that in the initial state, we ignore the location of the bird and assume it is on the slingshot. '''
    def create_pddl_problem(self, observation_array):

        pddl_problem = PddlPlusProblem()
        pddl_problem.domain = 'cartpole-plus-plus'
        pddl_problem.name = 'cartpole-plus-plus-prob'
        pddl_problem.metric = self.metric
        pddl_problem.objects = []
        pddl_problem.init = []
        pddl_problem.goal = []

        pddl_problem.objects.append(['dummy_obj', 'dummy'])
        pddl_problem.objects.append(['dummy_block', 'block'])


        euler_pole = self.quaternion_to_euler(round(observation_array['pole']['x_quaternion'], 5), round(observation_array['pole']['y_quaternion'], 5), round(observation_array['pole']['z_quaternion'], 5), round(observation_array['pole']['w_quaternion'], 5))
        obs_theta_x = round(euler_pole[0], 5) # XY reversed on purpose to match observation
        obs_theta_y = round(euler_pole[1], 5) # XY reversed on purpose to match observation
        # obs_theta_x = np.radians(round(observation_array['pole']['x_position'], 5))
        # obs_theta_y = np.radians(round(observation_array['pole']['y_position'], 5))

        obs_theta_x_dot = round(observation_array['pole']['x_velocity'], 5)
        obs_theta_y_dot = round(observation_array['pole']['y_velocity'], 5)
        obs_pos_x = round(observation_array['cart']['x_position'], 5)
        obs_pos_y = round(observation_array['cart']['y_position'], 5)
        obs_pos_x_dot = round(observation_array['cart']['x_velocity'], 5)
        obs_pos_y_dot = round(observation_array['cart']['y_velocity'], 5)


        # A dictionary with global problem parameters
        problem_params = dict()

        # Add constants fluents
        for numeric_fluent in self.constant_numeric_fluents:
            pddl_problem.init.append(['=', [numeric_fluent], round(self.constant_numeric_fluents[numeric_fluent],
                                                                   CartPolePlusPlusMetaModel.PLANNER_PRECISION)]) # TODO
        for boolean_fluent in self.constant_boolean_fluents:
            if self.constant_boolean_fluents[boolean_fluent]:
                pddl_problem.init.append([boolean_fluent])
            # else:
                # pddl_problem.init.append(['not',[boolean_fluent]])


        # MAIN COMPONENTS: X and THETA + derivatives
        pddl_problem.init.append(['=', ['pos_x'], round(obs_pos_x,CartPolePlusPlusMetaModel.PLANNER_PRECISION)])
        pddl_problem.init.append(['=', ['pos_y'], round(obs_pos_y, CartPolePlusPlusMetaModel.PLANNER_PRECISION)])
        pddl_problem.init.append(['=', ['pos_x_dot'], round(obs_pos_x_dot, CartPolePlusPlusMetaModel.PLANNER_PRECISION)])
        pddl_problem.init.append(['=', ['pos_y_dot'], round(obs_pos_y_dot, CartPolePlusPlusMetaModel.PLANNER_PRECISION)])
        pddl_problem.init.append(['=', ['theta_x'], round(obs_theta_x, CartPolePlusPlusMetaModel.PLANNER_PRECISION)])
        pddl_problem.init.append(['=', ['theta_y'], round(obs_theta_y, CartPolePlusPlusMetaModel.PLANNER_PRECISION)])
        pddl_problem.init.append(['=', ['theta_x_dot'], round(obs_theta_x_dot, CartPolePlusPlusMetaModel.PLANNER_PRECISION)])
        pddl_problem.init.append(['=', ['theta_y_dot'], round(obs_theta_y_dot, CartPolePlusPlusMetaModel.PLANNER_PRECISION)])
        pddl_problem.init.append(['=', ['F_x'], 10.0])
        pddl_problem.init.append(['=', ['F_y'], 0.0])

        initial_Fx = 10.0 # TODO: import initial force based on the last action applied, split initial_F into X and Y directions.
        initial_Fy = 0.0

        # TODO; WP: changed "self.constant_numeric_fluents['force_mag']" to "self.constant_numeric_fluents['F_x']" to "initial_F=0.0" at the beginning of calc_temp_x (verify that it's the correct thing to do).
        calc_temp_x = (initial_Fx + (self.constant_numeric_fluents['m_pole'] * self.constant_numeric_fluents['l_pole']) *
                     obs_theta_x_dot ** 2 * math.sin(obs_theta_x)) / (self.constant_numeric_fluents['m_cart'] + self.constant_numeric_fluents['m_pole'])
        calc_theta_x_ddot = (self.constant_numeric_fluents['gravity'] * math.sin(obs_theta_x) - math.cos(obs_theta_x) * calc_temp_x) / (self.constant_numeric_fluents['l_pole'] * (4.0 / 3.0 - self.constant_numeric_fluents['m_pole'] * math.cos(obs_theta_x) ** 2 / (self.constant_numeric_fluents['m_cart'] + self.constant_numeric_fluents['m_pole'])))
        calc_pos_x_ddot = calc_temp_x - (self.constant_numeric_fluents['m_pole'] * self.constant_numeric_fluents['l_pole']) * calc_theta_x_ddot * math.cos(obs_theta_x) / (self.constant_numeric_fluents['m_cart'] + self.constant_numeric_fluents['m_pole'])

        pddl_problem.init.append(['=', ['pos_x_ddot'], round(calc_pos_x_ddot, CartPolePlusPlusMetaModel.PLANNER_PRECISION)])
        pddl_problem.init.append(['=', ['theta_x_ddot'], round(calc_theta_x_ddot, CartPolePlusPlusMetaModel.PLANNER_PRECISION)])

        # TODO; WP: changed "self.constant_numeric_fluents['force_mag']" to "self.constant_numeric_fluents['F_y']" to "initial_F=0.0" at the beginning of calc_temp_y (verify that it's the correct thing to do).
        calc_temp_y = (initial_Fy + (self.constant_numeric_fluents['m_pole'] * self.constant_numeric_fluents['l_pole']) *
                     obs_theta_y_dot ** 2 * math.sin(obs_theta_y)) / (self.constant_numeric_fluents['m_cart'] + self.constant_numeric_fluents['m_pole'])
        calc_theta_y_ddot = (self.constant_numeric_fluents['gravity'] * math.sin(obs_theta_y) - math.cos(obs_theta_y) * calc_temp_y) / (self.constant_numeric_fluents['l_pole'] * (4.0 / 3.0 - self.constant_numeric_fluents['m_pole'] * math.cos(obs_theta_y) ** 2 / (self.constant_numeric_fluents['m_cart'] + self.constant_numeric_fluents['m_pole'])))
        calc_pos_y_ddot = calc_temp_y - (self.constant_numeric_fluents['m_pole'] * self.constant_numeric_fluents['l_pole']) * calc_theta_y_ddot * math.cos(obs_theta_y) / (self.constant_numeric_fluents['m_cart'] + self.constant_numeric_fluents['m_pole'])

        pddl_problem.init.append(['=', ['pos_y_ddot'], round(calc_pos_y_ddot, CartPolePlusPlusMetaModel.PLANNER_PRECISION)])
        pddl_problem.init.append(['=', ['theta_y_ddot'], round(calc_theta_y_ddot, CartPolePlusPlusMetaModel.PLANNER_PRECISION)])

        # Add goal
        # pddl_problem.goal.append(['pole_position'])
        pddl_problem.goal.append(['not', ['total_failure']])
        pddl_problem.goal.append(['=', ['elapsed_time'], ['time_limit']])

        return pddl_problem

    ''' Translate the given observed SBState to a PddlPlusState object.
    This is designed to handle intermediate state observed during execution '''
    def create_pddl_state(self, observations_array):
        pddl_state = PddlPlusState()

        # A dictionary with global problem parameters
        state_params = dict()


        euler_pole = self.quaternion_to_euler(round(observations_array['pole']['x_quaternion'], 5), round(observations_array['pole']['y_quaternion'], 5), round(observations_array['pole']['z_quaternion'], 5), round(observations_array['pole']['w_quaternion'], 5))

        obs_theta_x = round(euler_pole[0], 5)
        obs_theta_y = round(euler_pole[1], 5)
        obs_theta_x_dot = round(observations_array['pole']['x_velocity'], 5)
        obs_theta_y_dot = round(observations_array['pole']['y_velocity'], 5)

        obs_pos_x = round(observations_array['cart']['x_position'], 5)
        obs_pos_y = round(observations_array['cart']['y_position'], 5)
        obs_pos_x_dot = round(observations_array['cart']['x_velocity'], 5)
        obs_pos_y_dot = round(observations_array['cart']['y_velocity'], 5)

        state_params["pos_x"] = obs_pos_x
        state_params["pos_y"] = obs_pos_y
        state_params["pos_x_dot"] = obs_pos_x_dot
        state_params["pos_y_dot"] = obs_pos_y_dot
        state_params["theta_x"] = obs_theta_x
        state_params["theta_y"] = obs_theta_y
        state_params["theta_x_dot"] = obs_theta_x_dot
        state_params["theta_y_dot"] = obs_theta_y_dot

        for sp_key in state_params:
            pddl_state.numeric_fluents[tuple([sp_key])] = state_params[sp_key]

        return pddl_state

    def create_timed_action(self, action, time_step):
        ''' Create a PDDL+ TimedAction object from a world action and state '''

        if (action==1):
            action_name = "move_cart_right dummy_obj"
        elif (action==2):
            action_name = "move_cart_left dummy_obj"
        elif (action == 3):
            action_name = "move_cart_forward dummy_obj"
        elif (action == 4):
            action_name = "move_cart_backward dummy_obj"
        else:
            action_name = "do_nothing dummy_obj"

        action_time = time_step * 0.02

        return TimedAction(action_name, action_time)

    def quaternion_to_euler(self, x, y, z, w):
        ysqr = y * y

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + ysqr)
        X = np.arctan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = np.where(t2 > +1.0, +1.0, t2)
        # t2 = +1.0 if t2 > +1.0 else t2

        t2 = np.where(t2 < -1.0, -1.0, t2)
        # t2 = -1.0 if t2 < -1.0 else t2
        Y = np.arcsin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (ysqr + z * z)
        Z = np.arctan2(t3, t4)

        return (X, Y, Z)

