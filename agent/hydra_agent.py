#from agent.policy_learning.sarsa import SarsaLearner
from agent.consistency.model_formulation import ConsistencyChecker
from agent.planning.planner import Planner
import time

from agent.consistency.consistency_estimator import *
from worlds.science_birds_interface.client.agent_client import GameState
from agent.planning.pddl_meta_model import *

fh = logging.FileHandler("hydra.log",mode='w')
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger = logging.getLogger("hydra_agent")
logger.setLevel(logging.INFO)
logger.addHandler(fh)

class HydraAgent():
    '''
    Probably needs to subclass for each domain. We will cross that bridge when we get there
    '''

    def __init__(self,env=None):
        logger.info("[hydra_agent_server] :: Agent Created")
        self.env = env # agent always has a pointer to its environment
        if env is not None:
            env.sb_client.set_game_simulation_speed(settings.SB_SIM_SPEED)
        self.perception = Perception()
        self.consistency_checker = ConsistencyChecker()
        self.meta_model = MetaModel()
        self.planner = Planner(self.meta_model) # TODO: Discuss this w. Wiktor & Matt
        self.completed_levels = []
        self.observations = []
        self.novelty_likelihood = 0.0

    ''' Runs the agent. Returns False if the evaluation has not ended, and True if it has ended.'''
    def main_loop(self,max_actions=1000):
        logger.info("[hydra_agent_server] :: Entering main loop")
        logger.info("[hydra_agent_server] :: Delta t = {}".format(str(settings.DELTA_T)))
        logger.info("[hydra_agent_server] :: Simulation speed = {}\n\n".format(str(settings.SB_SIM_SPEED)))
        logger.info("[hydra_agent_server] :: Planner memory limit = {}".format(str(settings.PLANNER_MEMORY_LIMIT)))
        logger.info("[hydra_agent_server] :: Planner timeout = {}s\n\n".format(str(settings.TIMEOUT)))
        t = 0


        overall_plan_time = time.perf_counter()
        cumulative_plan_time = 0

        while t < max_actions:
            observation = ScienceBirdsObservation()  # Create an observation object to track on what happend
            raw_state = self.env.get_current_state()

            if raw_state.game_state.value == GameState.PLAYING.value:
                processed_state = self.perception.process_state(raw_state)
                observation.state = processed_state

                if processed_state and self.consistency_checker.is_consistent(processed_state):
                    logger.info("[hydra_agent_server] :: Invoking Planner".format())
                    orig_plan_time = time.perf_counter()
                    plan = self.planner.make_plan(processed_state, 0)
                    cumulative_plan_time += (time.perf_counter() - orig_plan_time)
                    logger.info("[hydra_agent_server] :: Original problem planning time: " + str((time.perf_counter() - orig_plan_time)))
                    if len(plan) == 0 or plan[0].action_name == "out of memory":
                        logger.info("[hydra_agent_server] :: Invoking Planner on a Simplified Problem {}".format('timeout' if len(plan)==0 else 'out of memory'))
                        simple_plan_time = time.perf_counter()
                        plan = self.planner.make_plan(processed_state, 2)
                        cumulative_plan_time += (time.perf_counter() - simple_plan_time)
                        logger.info("[hydra_agent_server] :: Simplified problem planning time: " + str((time.perf_counter() - simple_plan_time)))
                        if len(plan) == 0 or plan[0][0] == "out of memory": # TODO FIX THIS
                            plan = []
                            plan.append(self.__get_default_action(processed_state))

                    timed_action = plan[0]
                    logger.info("[hydra_agent_server] :: Taking action: {}".format(str(timed_action.action_name)))
                    sb_action = self.meta_model.create_sb_action(timed_action, processed_state)
                    raw_state, reward = self.env.act(sb_action)
                    observation.reward = reward
                    observation.action = sb_action
                    observation.intermediate_states = list(self.env.intermediate_states)
                    if settings.DEBUG:
                        observation.log_observation(self.planner.current_problem_prefix)
                    logger.info("[hydra_agent_server] :: Reward {} Game State {}".format(reward, raw_state.game_state))
                    # time.sleep(5)
                else:
                    logger.info("Perception Failure performing default shot")
                    plan = PddlPlusPlan()
                    plan.append(self.__get_default_action(processed_state))
                    sb_action = self.meta_model.create_sb_action(plan[0], processed_state)
                    raw_state, reward = self.env.act(sb_action)
                    logger.info("[hydra_agent_server] :: Reward {} Game State {}".format(reward, raw_state.game_state))
            elif raw_state.game_state.value == GameState.WON.value:
                self.completed_levels.append(True)
                logger.info("[hydra_agent_server] :: Level {} Complete - WIN".format(self.current_level))
                logger.info("[hydra_agent_server] :: Cumulative planning time only = {}".format(str(cumulative_plan_time)))
                logger.info("[hydra_agent_server] :: Planning effort percentage = {}\n".format(str( (cumulative_plan_time/(time.perf_counter() - overall_plan_time)) )))
                logger.info("[hydra_agent_server] :: Overall time to attempt level {} = {}\n\n".format(self.current_level, str((time.perf_counter() - overall_plan_time))))
                cumulative_plan_time = 0
                overall_plan_time = time.perf_counter()
                self.current_level = self.env.sb_client.load_next_available_level()
                # time.sleep(1)
                self.novelty_existence = self.env.sb_client.get_novelty_info()
                time.sleep(2/settings.SB_SIM_SPEED)
            elif raw_state.game_state.value == GameState.LOST.value:
                self.completed_levels.append(False)
                logger.info("[hydra_agent_server] :: Level {} complete - LOSS".format(self.current_level))
                logger.info("[hydra_agent_server] :: Cumulative planning time only = {}".format(str(cumulative_plan_time)))
                logger.info("[hydra_agent_server] :: Planning effort percentage = {}\n".format(str((cumulative_plan_time/(time.perf_counter() - overall_plan_time)))))
                logger.info("[hydra_agent_server] :: Overall time to attempt level {} = {}\n\n".format(self.current_level, str((time.perf_counter() - overall_plan_time))))
                cumulative_plan_time = 0
                overall_plan_time = time.perf_counter()
                self.current_level = self.env.sb_client.load_next_available_level()
                # time.sleep(1)
                self.novelty_existence = self.env.sb_client.get_novelty_info()
                time.sleep(2/settings.SB_SIM_SPEED)
            elif raw_state.game_state.value == GameState.NEWTRAININGSET.value:
                # DO something to start a fresh agent for a new training set
                (time_limit, interaction_limit, n_levels, attempts_per_level, mode, seq_or_set,
                 allowNoveltyInfo) = self.env.sb_client.ready_for_new_set()
                self.current_level = 0
                self.training_level_backup = 0
                change_from_training = True
                self.current_level = self.env.sb_client.load_next_available_level()
                self.novelty_existence = self.env.sb_client.get_novelty_info()
            elif raw_state.game_state.value == GameState.EVALUATION_TERMINATED.value:
                # store info and disconnect the agent as the evaluation is finished
                logger.info("Evaluation complete.")
                return True
            elif raw_state.game_state.value == GameState.REQUESTNOVELTYLIKELIHOOD.value:
                logger.info("[hydra_agent_server] :: Requesting Novelty Likelihood {}".format(self.consistency_checker.novelty_likelihood))
                # Require report novelty likelihood and then playing can be resumedconda env update -f environment.yml
                # dummy likelihoods:
                novelty_likelihood = self.consistency_checker.novelty_likelihood
                non_novelty_likelihood = 1 - novelty_likelihood
                self.env.sb_client.report_novelty_likelihood(novelty_likelihood, non_novelty_likelihood)
            elif raw_state.game_state.value == GameState.NEWTRIAL.value:
                # DO something to start a fresh agent for a new training set
                (time_limit, interaction_limit, n_levels, attempts_per_level, mode, seq_or_set,
                 allowNoveltyInfo) = self.env.sb_client.ready_for_new_set()
                logger.info("New Trial Request Received. Refresh agent.")
                self.current_level = 0
                self.training_level_backup = 0
                change_from_training = True
                self.current_level = self.env.sb_client.load_next_available_level()
                self.novelty_existence = self.env.sb_client.get_novelty_info()
            else:
                logger.info("[hydra_agent_server] :: Unexpected state.game_state.value {}".format(raw_state.game_state.value))
                assert False
            t+=1
            self.observations.append(observation)
        return False

    ''' A default action taken by the Hydra agent if planning fails'''
    def __get_default_action(self, state : ProcessedSBState):
        problem = self.meta_model.create_pddl_problem(state)
        pddl_state = PddlPlusState(problem.init)
        active_bird = pddl_state.get_active_bird()
        default_angle = 20.0
        default_time = self.meta_model.angle_to_action_time(default_angle, pddl_state)
        return TimedAction("pa-twang %s" % active_bird, default_time)

    def set_env(self,env):
        '''Probably bad to have two pointers here'''
        self.env = env

    ''' Runs the agent until it performs an action'''
    def run_next_action(self):
        while True:
            evaluation_done = self.main_loop(max_actions=1)
            if self.observations[-1].action is not None:
                return
            if evaluation_done == True:
                return


    ''' Finds the last observations of the game. That is, the last observation that has intermediate states. 
    TODO: Is this the best way to implement this?'''
    def find_last_obs(self):
        i = -1
        while self.observations[i].intermediate_states is None:
            i=i-1
        return self.observations[i]