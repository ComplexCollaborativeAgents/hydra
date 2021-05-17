'''
This module runs a set of experiments to evaluate the performance of our cartpole agent
'''

import logging
from os import path

import gym
import os.path as path
import settings
from agent.cartpole_hydra_agent import CartpoleHydraAgent, CartpoleHydraAgentObserver, RepairingCartpoleHydraAgent
from agent.planning.cartpole_pddl_meta_model import CartPoleMetaModel
from agent.repairing_hydra_agent import RepairingGymHydraAgent
import time
from agent.planning.cartpole_pddl_meta_model import *
from agent.gym_hydra_agent import *
from worlds.gym_cartpole_dispatcher import GymCartpoleDispatcher
from worlds.wsu.wsu_dispatcher import WSUObserver

logging.basicConfig(format='%(name)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("gym_experiment_runner")
logger.setLevel(logging.INFO)

''' Create Gym Environment '''
def _create_gym_env():
    return gym.make("CartPole-v1")

''' Run the given experiment type and output results to the results file '''
def _run_experiment(agent, experiment_type, result_file, agent_name = "Repairing", iterations = 3):
    hydra = agent
    iteration = 0
    while iteration < iterations:
        # Run the experiment
        start_time = time.time()
        hydra.run(False)
        runtime = time.time()-start_time

        # Collect data and print it to file
        observation = hydra.find_last_obs()
        iteration_reward = sum(observation.rewards)
        logger.info("Reward ! (%.2f), iteration %d" % (iteration_reward, iteration))
        result_file.write("%s\t %s\t %d\t %.2f\t %.2f\n" % (experiment_type, agent_name, iteration, iteration_reward, runtime))
        result_file.flush()

        iteration = iteration + 1
        hydra.observation = hydra.env.reset()

''' Run the Hydra agent with an oracle repair, i.e., modifying the meta_model params according to the injected fault'''
def _run_oracle_experiment(env_param, fluent_name, injected_faults = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]):
    result_file = open(path.join(settings.ROOT_PATH, "tests", "oracle_%s.csv" % env_param), "w")
    result_file.write("Fluent\t Value\t Run\t Agent\t Iteration\t Reward\t Runtime\n")
    env_nominal_value = CartPoleMetaModel().constant_numeric_fluents[fluent_name]
    repetitions = 5
    for fault_factor in injected_faults:
        env_param_value = env_nominal_value * fault_factor
        fluent_value = round(env_param_value,5) # TODO: Think about this more. This seems to be due to the planner misbehaving with too many decimal points

        for i in range(repetitions):
            env = _create_gym_env()

            # Inject fault
            env.env.__setattr__(env_param, env_param_value)

            # Perform perfect (oracle) repair action
            simple_agent = GymHydraAgent(env)
            simple_agent.meta_model.constant_numeric_fluents[fluent_name] = fluent_value # Perform the correct repair

            exp_name = "%s\t %s\t %s" % (env_param, fault_factor, i)
            _run_experiment(simple_agent, exp_name, result_file, agent_name="Normal")

    result_file.close()

''' Run the Hydra agent with an oracle repair, i.e., modifying the meta_model params according to the injected fault'''
def _run_repairing_experiment(env_param, fluent_name, injected_faults = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]):
    result_file = open(path.join(settings.ROOT_PATH, "tests", "repairing_%s.csv" % env_param), "w")
    result_file.write("Fluent\t Value\t Run\t Agent\t Iteration\t Reward\t Runtime\n")
    env_nominal_value = CartPoleMetaModel().constant_numeric_fluents[fluent_name]
    repetitions = 5
    for fault_factor in injected_faults:
        env_param_value = env_nominal_value * fault_factor
        for i in range(repetitions):
            env = _create_gym_env()

            # Inject fault
            env.env.__setattr__(env_param, env_param_value)

            # Run repairing agent
            repairing_agent = RepairingGymHydraAgent(env)

            exp_name = "%s\t %s\t %s" % (env_param, fault_factor, i)
            _run_experiment(repairing_agent, exp_name, result_file, agent_name="Repairing")

    result_file.close()

''' Run the Hydra agent without repairing. This checks the robustness to injected faults of the original model '''
def _run_no_repair_experiment(env_param, fluent_name,
                              injected_faults=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]):
    result_file = open(path.join(settings.ROOT_PATH, "tests", "repairing_%s.csv" % env_param), "w")
    result_file.write("Fluent\t Value\t Run\t Agent\t Iteration\t Reward\t Runtime\n")
    env_nominal_value = CartPoleMetaModel().constant_numeric_fluents[fluent_name]
    repetitions = 5
    for fault_factor in injected_faults:
        env_param_value = env_nominal_value * fault_factor
        for i in range(repetitions):
            env = _create_gym_env()

            # Inject fault
            env.env.__setattr__(env_param, env_param_value)

            # Run repairing agent
            repairing_agent = GymHydraAgent(env)

            exp_name = "%s\t %s\t %s" % (env_param, fault_factor, i)
            _run_experiment(repairing_agent, exp_name, result_file, agent_name="NoRepair")

    result_file.close()

''' Run experiments iwth the repairing Hydra agent '''
def run_repairing_agent_experiments(injected_faults = [0.8,1.0,1.2], env_param_to_fluent=dict(),
                                    run_experiment_funcs=[_run_repairing_experiment,_run_oracle_experiment,_run_no_repair_experiment]):
    for env_param in env_param_to_fluent.keys():
        fluent_name = env_param_to_fluent[env_param]
        for run_experiment_func in run_experiment_funcs:
            run_experiment_func(env_param, fluent_name, injected_faults=injected_faults)

#------------------ Experiments that are running via WSU's dispatcher/observer framework ------------------

''' Subclass of GymCartpoleDispatcher that injects a fault '''
class FaultyGymCartpoleDispatcher(GymCartpoleDispatcher):
    def __init__(self, delegate: WSUObserver, model_id: str = 'CartPole-v1', render: bool = False):
        super().__init__(delegate, model_id, render)
        self.faults = dict() # a dictionary of env parameter to value

    def inject_fault(self, env_parameter, env_value):
        self.faults[env_parameter]=env_value

    def _make_environment(self):
        env = gym.make(self.model_id)
        for env_param in self.faults:
            env_param_value = self.faults[env_param]
            env.env.__setattr__(env_param, env_param_value)
        return env

''' SUbclass of CartpoleHydraAgent that sets the correct fluent values to the agent '''
class CartpoleOracleHydraAgentObserver(CartpoleHydraAgent):
    def __init__(self, fluent_name, fluent_value):
        super().__init__(agent_type = CartpoleHydraAgent)
        self.fluent_name = fluent_name
        self.fluent_value = fluent_value

    def trial_start(self, trial_number: int, novelty_description: dict):
        super().trial_start(trial_number, novelty_description)
        self.agent.planner.meta_model.constant_numeric_fluents[self.fluent_name]= self.fluent_value

def run_repairing_observer_experiments(injected_faults = [0.8, 0.9, 1.0, 1.1, 1.2], env_param_to_fluent=dict()):
    ''' Run the Hydra agent with an oracle repair, i.e., modifying the meta_model params according to the injected fault'''
    for env_param in env_param_to_fluent.keys():
        fluent_name = env_param_to_fluent[env_param]
        result_file = open(path.join(settings.ROOT_PATH, "data", "cartpole", "repair", "consistency_repairing_%s_wsu.csv" % env_param), "w")
        result_file.write("Fluent\t Value\t Run\t Agent\t Iteration\t Nov.Likelihoods\t Consistency.Scores\t Rewards\t\t Avg.Reward\t Runtime\t Nov.Characterization\n")
        env_nominal_value = CartPoleMetaModel().constant_numeric_fluents[fluent_name]
        max_iterations = 5
        for fault_factor in injected_faults:
            env_param_value = env_nominal_value * fault_factor
            for i in range(max_iterations):
                observer = CartpoleHydraAgentObserver(agent_type=RepairingCartpoleHydraAgent)
                env = FaultyGymCartpoleDispatcher(observer, render=True)
                env.inject_fault(env_param, env_param_value)

                start_time = time.time()
                env.run()
                runtime = time.time() - start_time
                repair_performance = str(observer.agent.last_performance)
                exp_name = "%s\t %s\t %s" % (env_param, fault_factor, i)
                result_file.write(
                    "%s\t %s\t %d\t %s\t %s\t %s\t %.2f\t %.2f\t %s\n" % (exp_name, "Repairing", i, str(observer.agent.recorded_novelty_likelihoods), str(observer.agent.consistency_scores), repair_performance, sum(observer.agent.last_performance)/len(observer.agent.last_performance), runtime, observer.agent.novelty_characterization))
                result_file.flush()

                # # No repair
                # observer = CartpoleHydraAgentObserver(agent_type=CartpoleHydraAgent)
                # env = FaultyGymCartpoleDispatcher(observer, render=True)
                # env.inject_fault(env_param, env_param_value)
                #
                # start_time = time.time()
                # env.run()
                # runtime = time.time() - start_time
                # repair_performance = str(observer.agent.last_performance)
                # exp_name = "%s\t %s\t %s" % (env_param, fault_factor, i)
                # result_file.write(
                #     "%s\t %s\t %d\t %s\t %s\t %s\t %.2f\t %.2f\n" % (exp_name, " No repair", i, str(observer.agent.recorded_novelty_likelihoods), str(observer.agent.consistency_scores), repair_performance, sum(observer.agent.last_performance)/len(observer.agent.last_performance), runtime))
                # result_file.flush()


                # # Oracle
                # observer = CartpoleOracleHydraAgentObserver(env_param_to_fluent[env_param], round(env_param_value,
                #                                                                                   CartPoleMetaModel.PLANNER_PRECISION))
                # env = FaultyGymCartpoleDispatcher(observer, render=True)
                # env.inject_fault(env_param, env_param_value)
                #
                # start_time = time.time()
                # env.run()
                # runtime = time.time() - start_time
                # repair_performance = observer.agent.last_performance
                # exp_name = "%s\t %s\t %s" % (env_param, fault_factor, i)
                # result_file.write(
                #     "%s\t %s\t %d\t %.2f\t %.2f\n" % (exp_name, "Oracle", i, repair_performance, runtime))
                # result_file.flush()

        result_file.close()


if __name__ == '__main__':
    # Types of injected faults. 1.0 means no fault.
    injected_faults = [0.5, 1.0, 2.0]

    # Map fluent name to environment name
    env_param_to_fluent = dict()
    env_param_to_fluent['force_mag'] = 'force_mag'
    env_param_to_fluent['gravity'] = 'gravity'
    env_param_to_fluent['length'] = 'l_pole'
    env_param_to_fluent['masscart'] = 'm_cart'
    env_param_to_fluent['masspole'] = 'm_pole'
    env_param_to_fluent['x_threshold'] = 'x_limit'
    env_param_to_fluent['theta_threshold_radians'] = 'angle_limit'


    # Experiment types (these are functions that run an experiment
    # run_experiment_funcs = [_run_repairing_experiment, _run_oracle_experiment, _run_no_repair_experiment]
    # run_experiment_funcs = [_run_repairing_experiment]

    run_repairing_observer_experiments(injected_faults=injected_faults,
                                    env_param_to_fluent=env_param_to_fluent) # Run the experiment directly on the agent
