'''
This module runs a set of experiments to evaluate the performance of our cartpole agent
'''

import logging
import gym
import os.path as path
import settings
from agent.repairing_hydra_agent import RepairingGymHydraAgent
import time
from agent.planning.cartpole_pddl_meta_model import *
from agent.gym_hydra_agent import *

logging.basicConfig(format='%(name)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("gym_experiment_runner")
logger.setLevel(logging.INFO)

# Map fluent name to environment name
env_param_to_fluent = dict()
env_param_to_fluent['gravity'] = 'gravity'
env_param_to_fluent['force_mag'] = 'force_mag'
env_param_to_fluent['length'] = 'l_pole'
env_param_to_fluent['masscart'] = 'm_cart'
env_param_to_fluent['masspole'] = 'm_pole'
env_param_to_fluent['x_threshold'] = 'x_limit'
env_param_to_fluent['theta_threshold_radians'] = 'angle_limit'

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
def run_repairing_gym_experiments():
    injected_faults = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

    for env_param in env_param_to_fluent.keys():
        fluent_name = env_param_to_fluent[env_param]
        logging.info("")
        _run_repairing_experiment(env_param, fluent_name, injected_faults=injected_faults)

        _run_oracle_experiment(env_param, fluent_name, injected_faults=injected_faults)

        _run_no_repair_experiment(env_param, fluent_name, injected_faults=injected_faults)


if __name__ == '__main__':
    run_repairing_gym_experiments()