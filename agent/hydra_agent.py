from policy_learning.baseline import SarsaLearner
import logging

class HydraAgent():

    def __init__(self,env=None):
        logging.info("[hydra_agent_server] :: Agent Created")
        self.rl = SarsaLearner()
        self.env = env

    def main_loop(self):
        state = env.get_state()
