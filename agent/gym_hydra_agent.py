from agent.consistency.model_formulation import ConsistencyChecker
from agent.planning.cartpole_planner import CartpolePlanner
from agent.planning.cartpole_pddl_meta_model import *
import time

class GymHydraAgent:
    def __init__(self, env):
        self.env = env
        self.observation = self.env.reset()
        self.consistency_checker = ConsistencyChecker()
        self.meta_model = CartpoleMetaModel()
        self.cartpole_planner = CartpolePlanner(self.meta_model)
        self.novelty_likelihood = 0.0

    def run(self, max_actions=1000):

        plan = self.cartpole_planner.make_plan(self.observation, 0)
        print ("\nINITIAL STATE: ", self.observation)

        n_steps = 1

        for i in range(len(plan)):
            self.env.render()
            time.sleep(0.2)
            action = 1 if plan[i].action_name == "move_right" else 0
            self.observation, reward, done, info = self.env.step(action)
            print ("STEP: ", n_steps)
            print (action)
            print (self.observation)
            print (done)
            print ("\n")
            n_steps += 1

            if done:
                self.env.close()
                break
