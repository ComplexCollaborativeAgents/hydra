import math
from agent.planning.pddlplus_parser import *
import settings
import logging
from agent.planning.meta_model import *

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


class CartPoleMetaModel(MetaModel):
    PLANNER_PRECISION = 5 # how many decimal points the planner can handle correctly

    def __init__(self):
        super().__init__(
            docker_path=settings.CARTPOLE_PLANNING_DOCKER_PATH,
            domain_file_name="cartpole_domain.pddl",
            delta_t=settings.CP_DELTA_T,
            metric='minimize(total-time)',
            repairable_constants=('m_cart', 'l_pole', 'm_pole', 'force_mag', 'gravity', 'angle_limit', 'x_limit'),
            repair_deltas=(1.0, 0.1, 0.1, 1.0, 1.0, 0.1, 0.1),
            constant_numeric_fluents={
                'm_cart': 1.0,
                'friction_cart': 0.0,
                'l_pole': 0.5,
                'm_pole': 0.1,
                'friction_pole': 0.0,
                'force_mag': 10.0,
                'inertia': 1.0,
                'elapsed_time': 0.0,
                'gravity': 9.81,
                'time_limit': 1.0,
                'angle_limit': 0.205,
                'x_limit': 2.4},
            constant_boolean_fluents={
                'total_failure':False,
                'ready':True,
                'cart_available':True})

    ''' Translate the initial SBState, as observed, to a PddlPlusProblem object. 
    Note that in the initial state, we ignore the location of the bird and assume it is on the slingshot. '''
    def create_pddl_problem(self, observation_array):

        pddl_problem = PddlPlusProblem()
        pddl_problem.domain = 'cartpole-initial-bhaskara'
        pddl_problem.name = 'cartpole_prob'
        pddl_problem.metric = self.metric
        pddl_problem.objects = []
        pddl_problem.init = []
        pddl_problem.goal = []

        pddl_problem.objects.append(['dummy_obj', 'dummy'])

        obs_theta = round(observation_array[2], 5)
        obs_theta_dot = round(observation_array[3], 5)
        obs_x = round(observation_array[0], 5)
        obs_x_dot = round(observation_array[1], 5)


        # A dictionary with global problem parameters
        problem_params = dict()

        # Add constants fluents
        for numeric_fluent in self.constant_numeric_fluents:
            pddl_problem.init.append(['=', [numeric_fluent], round(self.constant_numeric_fluents[numeric_fluent],
                                                                   CartPoleMetaModel.PLANNER_PRECISION)]) # TODO
        for boolean_fluent in self.constant_boolean_fluents:
            if self.constant_boolean_fluents[boolean_fluent]:
                pddl_problem.init.append([boolean_fluent])
            # else:
                # pddl_problem.init.append(['not',[boolean_fluent]])


        # MAIN COMPONENTS: X and THETA + derivatives
        pddl_problem.init.append(['=', ['x'], round(obs_x,CartPoleMetaModel.PLANNER_PRECISION)])
        pddl_problem.init.append(['=', ['x_dot'], round(obs_x_dot, CartPoleMetaModel.PLANNER_PRECISION)])
        pddl_problem.init.append(['=', ['theta'], round(obs_theta, CartPoleMetaModel.PLANNER_PRECISION)])
        pddl_problem.init.append(['=', ['theta_dot'], round(obs_theta_dot, CartPoleMetaModel.PLANNER_PRECISION)])
        pddl_problem.init.append(['=', ['F'], round(self.constant_numeric_fluents['force_mag'], CartPoleMetaModel.PLANNER_PRECISION)])

        calc_temp = (self.constant_numeric_fluents['force_mag'] + (self.constant_numeric_fluents['m_pole'] *
                                                                   self.constant_numeric_fluents['l_pole']) *
                     obs_theta_dot ** 2 * math.sin(obs_theta)) / (self.constant_numeric_fluents['m_cart'] + self.constant_numeric_fluents['m_pole'])
        calc_theta_ddot = (self.constant_numeric_fluents['gravity'] * math.sin(obs_theta) - math.cos(obs_theta) * calc_temp) / (self.constant_numeric_fluents['l_pole'] * (4.0 / 3.0 - self.constant_numeric_fluents['m_pole'] * math.cos(obs_theta) ** 2 / (self.constant_numeric_fluents['m_cart'] + self.constant_numeric_fluents['m_pole'])))
        calc_x_ddot = calc_temp - (self.constant_numeric_fluents['m_pole'] * self.constant_numeric_fluents['l_pole']) * calc_theta_ddot * math.cos(obs_theta) / (self.constant_numeric_fluents['m_cart'] + self.constant_numeric_fluents['m_pole'])

        pddl_problem.init.append(['=', ['x_ddot'], round(calc_x_ddot, CartPoleMetaModel.PLANNER_PRECISION)])
        pddl_problem.init.append(['=', ['theta_ddot'], round(calc_theta_ddot, CartPoleMetaModel.PLANNER_PRECISION)])

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

        obs_theta = round(observations_array[2], 5)
        obs_theta_dot = round(observations_array[3], 5)
        obs_x = round(observations_array[0], 5)
        obs_x_dot = round(observations_array[1], 5)

        state_params["x"] = obs_x
        state_params["x_dot"] = obs_x_dot
        state_params["theta"] = obs_theta
        state_params["theta_dot"] = obs_theta_dot

        for sp_key in state_params:
            pddl_state.numeric_fluents[tuple([sp_key])] = state_params[sp_key]

        return pddl_state

    def create_timed_action(self, action, time_step):
        ''' Create a PDDL+ TimedAction object from a world action and state '''

        if (action==1):
            action_name = "move_cart_right dummy_obj"
        else:
            action_name = "move_cart_left dummy_obj"

        action_time = time_step * 0.02

        return TimedAction(action_name, action_time)

