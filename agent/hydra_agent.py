#from agent.policy_learning.sarsa import SarsaLearner
from agent.perception.perception import Perception
from agent.consistency.model_formulation import ConsistencyChecker
from agent.planning.planner import Planner
import worlds.science_birds as SB
import logging
import math
import time
import random

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


    def main_loop(self,max_actions=20,init_level=1):
        logger.info("[hydra_agent_server] :: Entering main loop")
        t = 0
        level = init_level
        # level = random.randint(11, 100)
        action = SB.SBLoadLevel(level)
        state, reward = self.env.act(action)
        while t < max_actions:
            tried_simplified_problem = False
            state = self.perception.process_state(state)
            if self.consistency_checker.is_consistent(state):
                if state.game_state.value == GameState.PLAYING.value:
                    logger.info("[hydra_agent_server] :: Invoking Planner".format())
                    init = time.perf_counter()
                    plan = self.planner.make_plan(state)
                    logger.info("planning time: " + str(time.perf_counter() - init))

                    if len(plan) == 0 or plan[0][0] == "out of memory":
                        logger.info("[hydra_agent_server] :: Invoking Planner on a Simplified Problem".format())
                        simplified_init = time.perf_counter()
                        plan = self.planner.make_plan(state,True)
                        logger.info("simplified planning time: " + str(time.perf_counter() -simplified_init))
                        if len(plan) == 0:
                            plan.append(("dummy-action: stab in the dark", 20.0))
                            logger.info("action: stab in the dark...")

                    logger.info("[hydra_agent_server] :: Taking action: {}".format(str(plan[0])))
                    ref_point = self.env.tp.get_reference_point(state.sling)
                    release_point_from_plan = \
                        self.env.tp.find_release_point(state.sling, math.radians(plan[0][1]))
                    action = SB.SBShoot(release_point_from_plan.X, release_point_from_plan.Y, 3000, ref_point.X,
                                         ref_point.Y)
                    state, reward = self.env.act(action)
                    logger.info("[hydra_agent_server] :: Reward {} Game State {}".format(reward, state.game_state))
                elif state.game_state.value == GameState.WON.value:
                    logger.info("[hydra_agent_server] :: Level {} complete".format(level))
                    level += 1
                    action = SB.SBLoadLevel(level)
                    state, reward = self.env.act(action)
                else: # move on to the next level
                    # assert False
                    logger.info("[hydra_agent_server] :: Level {} incomplete, moving on".format(level))
                    level += 1
                    action = SB.SBLoadLevel(level)
                    state, reward = self.env.act(action)
            else:
                assert False
            t+=1

    def set_env(self,env):
        '''Probably bad to have two pointers here'''
        self.env = env

