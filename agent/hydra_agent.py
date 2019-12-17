from agent.policy_learning.sarsa import SarsaLearner
from agent.perception.perception import Perception
from agent.consistency.model_formulation import ConsistencyChecker
from agent.planning.planner import Planner

import logging

class HydraAgent():

    def __init__(self,env=None):
        logging.info("[hydra_agent_server] :: Agent Created")
        self.env = env # agent always has a pointer to its environment
        self.perception = Perception()
        self.consistency_checker = ConsistencyChecker()
        self.planner = Planner()
        if env:
            self.rl = SarsaLearner(env)

    def main_loop(self,max_actions=20):
        # not sure how we want to replay actions (that is take more observations than actions)
        # is there a block on env.get_state()
        # there will need to be some additional configurations for planning is driving the process
        done = False
        t = 0
        while not done and t < max_actions:
            state = self.perception.process_state(self.env.get_current_state())
            if self.consistency_checker.is_consistent(state):
                plan = self.planner.make_plan(state)
                state, reward, done = self.planner.execute(plan,self.rl) # feels wierd to pass the rl agent in
            else:
                assert False
            t+=1

    def set_env(self,env):
        '''Probably bad to have two pointers here'''
        self.env = env
        self.rl.env = env

