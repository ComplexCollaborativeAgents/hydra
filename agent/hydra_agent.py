#from agent.policy_learning.sarsa import SarsaLearner
from agent.perception.perception import Perception
from agent.consistency.model_formulation import ConsistencyChecker
from agent.planning.planner import Planner
import worlds.science_birds as SB
import logging
import math
import time
import settings

from worlds.science_birds_interface.client.agent_client import GameState


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
        self.perception = Perception()
        self.consistency_checker = ConsistencyChecker()
        self.planner = Planner()
        self.novelty_likelihood = 0.0


    def main_loop(self,max_actions=1000):
        logger.info("[hydra_agent_server] :: Entering main loop")
        t = 0
        state = self.env.get_current_state()

        overall_plan_time = time.perf_counter()
        cumulative_plan_time = 0

        while t < max_actions:
            if state.game_state.value == GameState.PLAYING.value:
                state = self.perception.process_state(state)
                if state and self.consistency_checker.is_consistent(state):
                    logger.info("[hydra_agent_server] :: Invoking Planner".format())
                    settings.DELTA_T = 0.05
                    orig_plan_time = time.perf_counter()
                    plan = []
                    plan = self.planner.make_plan(state, 1)
                    cumulative_plan_time += (time.perf_counter() - orig_plan_time)
                    logger.info("[hydra_agent_server] :: Original problem planning time: " + str((time.perf_counter() - orig_plan_time)))
                    # plan = []
                    if len(plan) == 0 or plan[0][0] == "out of memory":
                        logger.info("[hydra_agent_server] :: Invoking Planner on a Simplified Problem".format())
                        settings.DELTA_T = 0.05
                        simple_plan_time = time.perf_counter()
                        plan = self.planner.make_plan(state, 2)
                        cumulative_plan_time += (time.perf_counter() - simple_plan_time)
                        logger.info("[hydra_agent_server] :: Simplified problem planning time: " + str((time.perf_counter() - simple_plan_time)))
                        if len(plan) == 0 or plan[0][0] == "out of memory":
                            plan.append(("dummy-action", 20.0))
                    logger.info("[hydra_agent_server] :: Taking action: {}".format(str(plan[0])))
                    ref_point = self.env.tp.get_reference_point(state.sling)
                    release_point_from_plan = \
                        self.env.tp.find_release_point(state.sling, math.radians(plan[0][1]))
                    action = SB.SBShoot(release_point_from_plan.X, release_point_from_plan.Y, 3000, ref_point.X,
                                         ref_point.Y)
                    state, reward = self.env.act(action)
                    logger.info("[hydra_agent_server] :: Reward {} Game State {}".format(reward, state.game_state))
                    # time.sleep(5)
                else:
                    logger.info("Perception Failure performing default shot")
                    plan = []
                    plan.append(("dummy-action", 20.0))
                    ref_point = self.env.tp.get_reference_point(state.sling)
                    release_point_from_plan = \
                        self.env.tp.find_release_point(state.sling, math.radians(plan[0][1]))
                    action = SB.SBShoot(release_point_from_plan.X, release_point_from_plan.Y, 3000, ref_point.X, ref_point.Y)
                    state, reward = self.env.act(action)
                    logger.info("[hydra_agent_server] :: Reward {} Game State {}".format(reward, state.game_state))
            elif state.game_state.value == GameState.WON.value:
                logger.info("[hydra_agent_server] :: Level {} Complete - WIN".format(self.current_level))
                logger.info("[hydra_agent_server] :: Cumulative planning time only = {}".format(str(cumulative_plan_time)))
                logger.info("[hydra_agent_server] :: Planning effort percentage = {}\n".format(str( (cumulative_plan_time/(time.perf_counter() - overall_plan_time)) )))
                logger.info("[hydra_agent_server] :: Overall time to attempt level {} = {}\n\n".format(self.current_level, str((time.perf_counter() - overall_plan_time))))
                cumulative_plan_time = 0
                overall_plan_time = time.perf_counter()
                self.current_level = self.env.sb_client.load_next_available_level()
                #self.novelty_existence = self.env.sb_client.get_novelty_info()
                # time.sleep(2)
                state = self.env.get_current_state()
            elif state.game_state.value == GameState.LOST.value:
                logger.info("[hydra_agent_server] :: Level {} complete - LOSS".format(self.current_level))
                logger.info("[hydra_agent_server] :: Cumulative planning time only = {}".format(str(cumulative_plan_time)))
                logger.info("[hydra_agent_server] :: Planning effort percentage = {}\n".format(str((cumulative_plan_time/(time.perf_counter() - overall_plan_time)))))
                logger.info("[hydra_agent_server] :: Overall time to attempt level {} = {}\n\n".format(self.current_level, str((time.perf_counter() - overall_plan_time))))
                cumulative_plan_time = 0
                overall_plan_time = time.perf_counter()
                self.current_level = self.env.sb_client.load_next_available_level()
                #self.novelty_existence = self.env.sb_client.get_novelty_info()
                # time.sleep(2)
                state = self.env.get_current_state()
            elif state.game_state.value == GameState.NEWTRAININGSET.value:
                # DO something to start a fresh agent for a new training set
                (time_limit, interaction_limit, n_levels, attempts_per_level, mode, seq_or_set,
                 allowNoveltyInfo) = self.env.sb_client.ready_for_new_set()
                self.current_level = 0
                self.training_level_backup = 0
                change_from_training = True
                self.current_level = self.env.sb_client.load_next_available_level()
                #self.novelty_existence = self.env.sb_client.get_novelty_info()
                state = self.env.get_current_state()
            elif state.game_state.value == GameState.EVALUATION_TERMINATED.value:
                # store info and disconnect the agent as the evaluation is finished
                logger.info("Evaluation complete.")
                return None
            elif state.game_state.value == GameState.REQUESTNOVELTYLIKELIHOOD.value:
                logger.info("[hydra_agent_server] :: Requesting Novelty Likelihood {}".format(0.1))
                # Require report novelty likelihood and then playing can be resumed
                # dummy likelihoods:
                novelty_likelihood = self.consistency_checker.novelty_likelihood
                non_novelty_likelihood = 1 - novelty_likelihood
                self.env.sb_client.report_novelty_likelihood(novelty_likelihood, non_novelty_likelihood)
                state = self.env.get_current_state()
            elif state.game_state.value == GameState.NEWTRIAL.value:
                # DO something to start a fresh agent for a new training set
                (time_limit, interaction_limit, n_levels, attempts_per_level, mode, seq_or_set,
                 allowNoveltyInfo) = self.env.sb_client.ready_for_new_set()
                logger.info("New Trial Request Received. Refresh agent.")
                self.current_level = 0
                self.training_level_backup = 0
                change_from_training = True
                self.current_level = self.env.sb_client.load_next_available_level()
                #self.novelty_existence = self.env.sb_client.get_novelty_info()
                state = self.env.get_current_state()
            else:
                logger.info("[hydra_agent_server] :: Unexpected state.game_state.value {}".format(state.game_state.value))
                assert False
            t+=1

    def set_env(self,env):
        '''Probably bad to have two pointers here'''
        self.env = env

