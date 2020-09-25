import settings
from agent.repairing_hydra_agent import RepairingHydraSBAgent,RepairingGymHydraAgent
from agent.gym_hydra_agent import *
import pytest
from agent.planning.pddl_meta_model import *
import gym
import subprocess
import worlds.science_birds as sb
import pickle
import tests.test_utils as test_utils
import os.path as path
import time



GRAVITY_FACTOR = "gravity_factor"
DATA_DIR = path.join(settings.ROOT_PATH, 'data')
TEST_DATA_DIR = path.join(DATA_DIR, 'science_birds', 'tests')

logger = test_utils.create_logger("test_repairing_agent")

#################### System tests ########################
@pytest.fixture(scope="module")
def launch_science_birds():
    logger.info("Starting ScienceBirds")
    cmd = 'cp {}/data/science_birds/level-04.xml {}/00001.xml'.format(str(settings.ROOT_PATH), str(settings.SCIENCE_BIRDS_LEVELS_DIR))
    subprocess.run(cmd, shell=True)

    env = sb.ScienceBirds(None,launch=True,config='test_consistency_config.xml')
    yield env
    env.kill()
    logger.info("Ending ScienceBirds")

''' Inject a fault to the agent's meta model '''
def _inject_fault_to_sb_meta_model(meta_model : MetaModel, fluent_to_change = GRAVITY_FACTOR):
    meta_model.constant_numeric_fluents[fluent_to_change] = 6.0


''' A full system test: run SB with a bad meta model, observe results, fix meta model '''
@pytest.mark.skip("Have not migrated to 0.3.6 yet")
def test_repair_gravity_in_sb_agent(launch_science_birds):
    # Setup environment and agent
    save_obs = True
    plot_exp_vs_obs = True

    settings.SB_SIM_SPEED=5
    settings.SB_GT_FREQ =1

    env = launch_science_birds
    hydra = RepairingHydraSBAgent(env)

    # Inject fault and run the agent
    _inject_fault_to_sb_meta_model(hydra.meta_model, GRAVITY_FACTOR)

    iteration = 0
    obs_with_rewards = 0

    while iteration < 4:
        hydra.run_next_action()
        observation = hydra.find_last_obs()

        # Store observation for debug
        if save_obs:
            obs_output_file = path.join(TEST_DATA_DIR, "test_repair_gravity_in_agent_obs_%d.p" % iteration)  # For debug
            pickle.dump(observation, open(obs_output_file, "wb"))  # For debug
        if plot_exp_vs_obs:
            test_utils.plot_expected_vs_observed(hydra.meta_model, observation)

        if observation.reward > 0:
            logger.info("Reward ! (%.2f), iteration %d" % (observation.reward, iteration))
            obs_with_rewards = obs_with_rewards + 1
        iteration=iteration+1

    assert obs_with_rewards>0

#### CARTPOLE TESTS ####
@pytest.fixture(scope="module")
def launch_cartpole():
    logger.info("Starting CartPole")
    env = gym.make("CartPole-v1")
    yield env
    # env.kill()
    logger.info("Ending CartPole")

''' Inject a fault to the cartpole environment '''
def _inject_fault_to_carpole_env(env):
    env.env.gravity=13


''' A full system test: run SB with a bad meta model, observe results, fix meta model '''
def test_repair_gravity_in_cartpole_agent(launch_cartpole):
    # Setup environment and agent
    save_obs = True

    result_file = open(path.join(settings.ROOT_PATH,"tests", "repair_gravity_13.csv"),"w")
    result_file.write("Agent\t Iteration\t Reward\n")

    env = launch_cartpole
    hydra = RepairingGymHydraAgent(env)
    agent_name = "Repairing"

    _inject_fault_to_carpole_env(env)

    max_iterations = 4
    iteration = 0
    while iteration < max_iterations:
        hydra.run()
        observation = hydra.find_last_obs()
        iteration_reward=sum(observation.rewards)
        logger.info("Reward ! (%.2f), iteration %d" % (iteration_reward, iteration))

        # Store observation for debug
        if save_obs:
            obs_output_file = path.join(TEST_DATA_DIR, "test_repair_gravity_in_cartpole_agent_obs_%d.p" % iteration)  # For debug
            pickle.dump(observation, open(obs_output_file, "wb"))  # For debug
            obs_output_file = path.join(TEST_DATA_DIR, "test_repair_gravity_in_cartpole_agent_obs_%d.mm" % iteration)  # For debug
            pickle.dump(hydra.meta_model, open(obs_output_file, "wb"))  # For debug


        result_file.write("%s\t %d\t %.2f\n" % (agent_name, iteration, iteration_reward))
        result_file.flush()
        iteration = iteration+1
        hydra.observation = hydra.env.reset()


    hydra = GymHydraAgent(env)
    agent_name = "Vanilla"
    iteration = 0
    while iteration < max_iterations:
        hydra.run()
        observation = hydra.find_last_obs()
        iteration_reward=sum(observation.rewards)
        logger.info("Reward ! (%.2f), iteration %d" % (iteration_reward, iteration))

        # Store observation for debug
        if save_obs:
            obs_output_file = path.join(TEST_DATA_DIR, "test_repair_gravity_in_cartpole_agent_obs_%d.p" % iteration)  # For debug
            pickle.dump(observation, open(obs_output_file, "wb"))  # For debug

        result_file.write("%s\t %d\t %.2f\n" % (agent_name, iteration, iteration_reward))
        result_file.flush()
        iteration = iteration+1

    result_file.close()



''' Run a suite of experiments '''
def test_repairing_gym_experiments():
    # Setup environment and agent
    save_obs = True

    result_file = open(path.join(settings.ROOT_PATH, "tests", "repair_gravity5-10.csv"), "w")
    result_file.write("FaultType\tAgent\t Iteration\t Reward\t Runtime\n")


    gravity_values = [5,6,7,8,9,10,11,12,13,14,15]
    for gravity in gravity_values:
        env =  gym.make("CartPole-v1")
        env.env.gravity=gravity
        fault_type = "gravity-%d" % gravity
        _run_experiment(env, fault_type, result_file)

    result_file.close()

def _run_experiment(env, fault_type, result_file):
    max_iterations = 2

    agent_name = "Repairing"
    hydra = RepairingGymHydraAgent(env)
    iteration = 0
    while iteration < max_iterations:
        start_time = time.time()
        hydra.run()
        runtime = time.time()-start_time
        observation = hydra.find_last_obs()
        iteration_reward = sum(observation.rewards)
        logger.info("Reward ! (%.2f), iteration %d" % (iteration_reward, iteration))

        result_file.write("%s\t %s\t %d\t %.2f\t %.2f\n" % (fault_type, agent_name, iteration, iteration_reward, runtime))
        result_file.flush()
        iteration = iteration + 1
        hydra.observation = hydra.env.reset()

    agent_name = "Vanilla"
    hydra = GymHydraAgent(env)
    iteration = 0
    while iteration < max_iterations:
        start_time = time.time()
        hydra.run()
        runtime = time.time()-start_time
        observation = hydra.find_last_obs()
        iteration_reward = sum(observation.rewards)
        logger.info("Reward ! (%.2f), iteration %d" % (iteration_reward, iteration))
        result_file.write("%s\t %s\t %d\t %.2f\t %.2f\n" % (fault_type, agent_name, iteration, iteration_reward, runtime))
        result_file.flush()
        iteration = iteration + 1
        hydra.observation = hydra.env.reset()





''' Run a suite of experiments '''
def test_repairing_gym_experiments2():
    # Setup environment and agent
    save_obs = True

    result_file = open(path.join(settings.ROOT_PATH, "tests", "repair_gravity-all.csv"), "w")
    result_file.write("Gravity\tRepair params\tAgent\t Iteration\t Reward\t Runtime\n")


    gravity_values = [5,6,7,8,9,10,11,12,13,14,15]
    repairable_constants = ('gravity', 'm_cart', 'friction_cart', 'l_pole', 'm_pole',)
    repair_deltas = (1.0, 0.5, 0.5, 0.25, 0.1, 0.2, 1.0,)
    for gravity in gravity_values:
        env =  gym.make("CartPole-v1")
        env.env.gravity=gravity
        for i in range(1,len(repairable_constants)):
            exp_name = "%d\t %d" % (gravity, i)

            baseline_agent = GymHydraAgent(env)
            repairing_agent = RepairingGymHydraAgent(env)
            repairing_agent.meta_model.repairable_constants = repairable_constants[:i]
            repairing_agent.meta_model.repair_deltas = repair_deltas[:i]

            _run_experiment2(baseline_agent, repairing_agent, exp_name, result_file)

    result_file.close()

def _run_experiment2(baseline_agent, repairing_agent, experiment_type, result_file):
    max_iterations = 2

    agent_name = "Repairing"
    hydra = repairing_agent
    iteration = 0
    while iteration < max_iterations:
        start_time = time.time()
        hydra.run()
        runtime = time.time()-start_time
        observation = hydra.find_last_obs()
        iteration_reward = sum(observation.rewards)
        logger.info("Reward ! (%.2f), iteration %d" % (iteration_reward, iteration))

        result_file.write("%s\t %s\t %d\t %.2f\t %.2f\n" % (experiment_type, agent_name, iteration, iteration_reward, runtime))
        result_file.flush()
        iteration = iteration + 1
        hydra.observation = hydra.env.reset()

    agent_name = "Vanilla"
    hydra = baseline_agent
    iteration = 0
    while iteration < max_iterations:
        start_time = time.time()
        hydra.run()
        runtime = time.time()-start_time
        observation = hydra.find_last_obs()
        iteration_reward = sum(observation.rewards)
        logger.info("Reward ! (%.2f), iteration %d" % (iteration_reward, iteration))
        result_file.write("%s\t %s\t %d\t %.2f\t %.2f\n" % (experiment_type, agent_name, iteration, iteration_reward, runtime))
        result_file.flush()
        iteration = iteration + 1
        hydra.observation = hydra.env.reset()



