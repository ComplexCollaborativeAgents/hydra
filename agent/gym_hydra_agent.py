from agent.consistency.model_formulation import ConsistencyChecker
from agent.planning.cartpole_planner import CartPolePlanner
from agent.planning.cartpole_pddl_meta_model import *
from agent.consistency.observation import CartPoleObservation
import time
import copy
import numpy as np
import settings
import matplotlib.pyplot as plt
import random

MIN_STEPS_TO_REPLAN = 40

class GymHydraAgent:
    def __init__(self, env, starting_seed=False):
        self.env = env
        self.observation = self.env.reset()
        if starting_seed == True:
            self.observation = self.reset_with_seed()
        self.meta_model = CartPoleMetaModel()
        self.cartpole_planner = CartPolePlanner(self.meta_model)
        self.novelty_likelihood = 0.0
        self.observations_list = []

    def run(self, debug_info=False, max_actions=1000):
        self.meta_model.constant_numeric_fluents['time_limit'] = 4.0
        # print("TIME_LIMIT", self.meta_model.constant_numeric_fluents['time_limit'])
        plan = self.cartpole_planner.make_plan(self.observation, 0)
        # print("\n\nPLAN LENGTH: ", len(plan))
        if debug_info:
            print ("\nINITIAL STATE: ", self.observation)
            print("GYM INITIAL STATE: ", self.env.state)

        n_steps = 1
        cartpole_obs = CartPoleObservation()

        if debug_info:
            initial_state_exec = (self.observation[0], self.observation[1], self.observation[2], self.observation[3], 0.00)
            state_values_list = (self.cartpole_planner.extract_state_values_from_trace("%s/plan_cartpole_prob.pddl" % str(settings.CARTPOLE_PLANNING_DOCKER_PATH)))
            # state_values_list.insert(0, initial_state_exec)
            full_plan_trace = []
            # full_plan_trace.append(initial_state_exec)
        emergency_plan = False

        itt = 0
        while True:
            self.env.render()
            time.sleep(0.05)
            if itt<len(plan):
                action = 1 if plan[itt].action_name == "move_cart_right dummy_obj" else 0
            else:
                action = random.randint(0,1) # Choose random action if plan failed
            cartpole_obs.actions.append(action)
            cartpole_obs.states.append(self.observation)
            self.observation, reward, done, info = self.env.step(action)
            cartpole_obs.rewards.append(reward)

            # cartpole_obs.states.append(self.observation)
            if debug_info:
                full_plan_trace.append(state_values_list[itt])
                print ("\nSTEP: ", n_steps, str(round(n_steps*0.02,4)) + "s")
                print (action)
                print (self.observation)
                print (full_plan_trace[-1])
                # print ((full_plan_trace[-2][0],full_plan_trace[-1][1],full_plan_trace[-2][2],full_plan_trace[-1][3])) if len(full_plan_trace) > 1 else print(full_plan_trace[-1])
                print ("REWARD:", reward)
                print (done)

            n_steps += 1
            itt += 1

            if done or n_steps >= 201:
                print ("\n\nFINISHED\nSCORE: ", sum(cartpole_obs.rewards))
                self.env.close()
                break

            if (itt >= MIN_STEPS_TO_REPLAN):
                # print (n_steps)
                emergency_plan = False

                temp_plan = copy.copy(plan)
                self.meta_model.constant_numeric_fluents['time_limit'] = round((4.0 - ((n_steps-1)*0.02)), 2)
                plan = self.cartpole_planner.make_plan(self.observation, 0)
                if (len(plan)) == 0:
                    emergency_plan = True
                    plan = temp_plan
                    print("\n\nempty plan, reusing extra actions from previous plan...\n")
                    continue
                    # plan = self.cartpole_planner.make_plan(self.observation, 0)

                if debug_info:
                    state_values_list = (self.cartpole_planner.extract_state_values_from_trace("%s/plan_cartpole_prob.pddl" % str(settings.CARTPOLE_PLANNING_DOCKER_PATH)))

                itt = 0

        if debug_info:
            full_plan_trace.insert(0, initial_state_exec)

        # DOES NOT INCLUDE THE FINAL STATE (i.e. GOAL STATE)
        self.observations_list.append(cartpole_obs)
        if debug_info:
            print ("\n\nCARTPOLE OBSERVATIONS")
            print (cartpole_obs.states)
            print (cartpole_obs.actions)
            print (sum(cartpole_obs.rewards))
            self.plot_plan_vs_execution(full_plan_trace, cartpole_obs, n_steps)

    def reset_with_seed(self):
        self.env.state = self.env.np_random.uniform(low=0.02, high=0.02, size=(4,))
        self.env.steps_beyond_done = None
        return np.array(self.env.state)

    def find_last_obs(self):
        if len(self.observations_list)==0:
            return None
        else:
            return self.observations_list[-1]

    def plot_plan_vs_execution(self, plan_vals, exec_vals : CartPoleObservation, steps):

        plan_xs = []
        plan_x_dots = []
        plan_thetas = []
        plan_theta_dots = []

        exec_xs = []
        exec_x_dots = []
        exec_thetas = []
        exec_theta_dots = []

        for j in range(len(exec_vals.states)):
            exec_xs.append(exec_vals.states[j][0])
            exec_x_dots.append(exec_vals.states[j][1])
            exec_thetas.append(exec_vals.states[j][2])
            exec_theta_dots.append(exec_vals.states[j][3])

            plan_xs.append(plan_vals[j][0])
            plan_x_dots.append(plan_vals[j][1])
            plan_thetas.append(plan_vals[j][2])
            plan_theta_dots.append(plan_vals[j][3])

        plt.title('Cart Position (X)')
        plt.plot(np.arange(1,steps,1), exec_xs, label='exec')
        plt.plot(np.arange(1,steps,1), plan_xs, label='plan')
        plt.xlabel('steps')
        plt.xticks(np.arange(0, steps, 40))
        plt.ylabel('values')
        plt.legend()
        plt.show()

        plt.title('Cart Velocity (X dot)')
        plt.plot(np.arange(1, steps, 1), exec_x_dots, label='exec')
        plt.plot(np.arange(1, steps, 1), plan_x_dots, label='plan')
        plt.xlabel('steps')
        plt.xticks(np.arange(0, steps, 40))
        plt.ylabel('values')
        plt.legend()
        plt.show()

        plt.title('Pole Position (Theta)')
        plt.plot(np.arange(1, steps, 1), exec_thetas, label='exec')
        plt.plot(np.arange(1, steps, 1), plan_thetas, label='plan')
        plt.xlabel('steps')
        plt.xticks(np.arange(0, steps, 40))
        plt.ylabel('values')
        plt.legend()
        plt.show()

        plt.title('Pole Velocity (Theta dot)')
        plt.plot(np.arange(1, steps, 1), exec_theta_dots, label='exec')
        plt.plot(np.arange(1, steps, 1), plan_theta_dots, label='plan')
        plt.xlabel('steps')
        plt.xticks(np.arange(0, steps, 40))
        plt.ylabel('values')
        plt.legend()
        plt.show()