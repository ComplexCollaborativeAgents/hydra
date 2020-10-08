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

        # Store observation and meta model for debugging
        if save_obs:
            obs_output_file = path.join(TEST_DATA_DIR, "test_repair_gravity_in_cartpole_agent_obs_%d.p" % iteration)  # For debug
            pickle.dump(observation, open(obs_output_file, "wb"))  # For debug
            obs_output_file = path.join(TEST_DATA_DIR, "test_repair_gravity_in_cartpole_agent_obs_%d.mm" % iteration)  # For debug
            pickle.dump(hydra.meta_model, open(obs_output_file, "wb"))  # For debug


        result_file.write("%s\t %d\t %.2f\n" % (agent_name, iteration, iteration_reward))
        result_file.flush()
        iteration = iteration+1
        hydra.observation = hydra.env.reset()

''' Run a suite of experiments '''
def test_repairing_gym_experiments():
    result_file = open(path.join(settings.ROOT_PATH, "tests", "repair_gravity-all-big.csv"), "w")
    result_file.write("Gravity\t Repair params\t Run\t Agent\t Iteration\t Reward\t Runtime\n")

    max_iterations = 5
    gravity_values = [5,6,7,8,9,10,11,12,13,14,15]
    repairable_constants = ('gravity', 'm_cart', 'friction_cart', 'l_pole', 'm_pole',)
    repair_deltas = (1.0, 0.5, 0.5, 0.25, 0.1, 0.2, 1.0,)
    for gravity in gravity_values:
        env =  gym.make("CartPole-v1")
        env.env.gravity=gravity
        for i in range(1,len(repairable_constants)):
            for j in range(max_iterations):
                exp_name = "%d\t %d\t %d" % (gravity, i,j)

                repairing_agent = RepairingGymHydraAgent(env)
                repairing_agent.meta_model.repairable_constants = repairable_constants[:i]
                repairing_agent.meta_model.repair_deltas = repair_deltas[:i]

                _run_experiment(repairing_agent, exp_name, result_file)

    result_file.close()

''' Run the Hydra agent with an oracle repair, i.e., modifying the meta_model params according to the injected fault'''
def test_model_robustness():
    env_param_to_fluent = dict()
    env_param_to_fluent['gravity'] = 'gravity'
    env_param_to_fluent['force_mag'] = 'force_mag'
    env_param_to_fluent['length'] = 'l_pole'
    env_param_to_fluent['masscart'] = 'm_cart'
    env_param_to_fluent['masspole'] = 'm_pole'
    env_param_to_fluent['x_threshold'] = 'x_limit'
    env_param_to_fluent['theta_threshold_radians'] = 'angle_limit'

    for env_param in env_param_to_fluent.keys():
        fluent_name = env_param_to_fluent[env_param]
        _run_oracle_experiment(env_param, fluent_name)

''' Run the Hydra agent with an oracle repair, i.e., modifying the meta_model params according to the injected fault'''
def test_model_repair():
    env_param_to_fluent = dict()
    env_param_to_fluent['gravity'] = 'gravity'
    env_param_to_fluent['force_mag'] = 'force_mag'
    env_param_to_fluent['length'] = 'l_pole'
    env_param_to_fluent['masscart'] = 'm_cart'
    env_param_to_fluent['masspole'] = 'm_pole'
    env_param_to_fluent['x_threshold'] = 'x_limit'
    env_param_to_fluent['theta_threshold_radians'] = 'angle_limit'

    for env_param in env_param_to_fluent.keys():
        fluent_name = env_param_to_fluent[env_param]
        _run_repairing_experiment(env_param, fluent_name)


def test_oracle_force_mag():
    _run_oracle_experiment("force_mag", "force_mag")

def test_oracle_gravity():
    _run_oracle_experiment("gravity", "gravity")

def test_oracle_l_pole():
    _run_oracle_experiment("length", "l_pole")

def test_oracle_m_cart():
    _run_oracle_experiment("masscart", "m_cart")

def test_oracle_m_pole():
    _run_oracle_experiment("masspole", "m_pole")

def test_oracle_x_limit():
    _run_oracle_experiment("x_threshold", "x_limit")

def test_oracle_angle_limit():
    _run_oracle_experiment("angle_limit", "theta_threshold_radians")


def test_repairing_force_mag():
    _run_repairing_experiment("force_mag", "force_mag")

def test_repairing_gravity():
    _run_repairing_experiment("gravity", "gravity")

def test_repairing_l_pole():
    _run_repairing_experiment("length", "l_pole")

def test_repairing_m_cart():
    _run_oracle_experiment("masscart", "m_cart")

def test_repairing_m_pole():
    _run_repairing_experiment("masspole", "m_pole")

def test_repairing_x_limit():
    _run_repairing_experiment("x_threshold", "x_limit")

def test_repairing_angle_limit():
    _run_repairing_experiment("angle_limit", "theta_threshold_radians")

''' Run the given experiment type and output results to the results file '''
def _run_experiment(agent, experiment_type, result_file, agent_name = "Repairing"):
    max_iterations = 3
    hydra = agent
    iteration = 0
    while iteration < max_iterations:
        start_time = time.time()
        hydra.run(False)
        runtime = time.time()-start_time
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
    max_iterations = 5
    for fault_factor in injected_faults:
        env_param_value = env_nominal_value * fault_factor
        fluent_value = env_param_value

        for i in range(max_iterations):
            env = gym.make("CartPole-v1")
            env.env.__setattr__(env_param, env_param_value)
            exp_name = "%s\t %s\t %s" % (env_param, fault_factor, i)
            simple_agent = GymHydraAgent(env)
            simple_agent.meta_model.constant_numeric_fluents[fluent_name] = fluent_value # Perform the correct repair
            _run_experiment(simple_agent, exp_name, result_file, agent_name="Normal")

    result_file.close()

''' Run the Hydra agent with an oracle repair, i.e., modifying the meta_model params according to the injected fault'''
def _run_repairing_experiment(env_param, fluent_name, injected_faults = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]):
    result_file = open(path.join(settings.ROOT_PATH, "tests", "repairing_%s.csv" % env_param), "w")
    result_file.write("Fluent\t Value\t Run\t Agent\t Iteration\t Reward\t Runtime\n")
    env_nominal_value = CartPoleMetaModel().constant_numeric_fluents[fluent_name]
    max_iterations = 5
    for fault_factor in injected_faults:
        env_param_value = env_nominal_value * fault_factor
        for i in range(max_iterations):
            env = gym.make("CartPole-v1")
            env.env.__setattr__(env_param, env_param_value)
            exp_name = "%s\t %s\t %s" % (env_param, fault_factor, i)

            repairing_agent = RepairingGymHydraAgent(env)
            _run_experiment(repairing_agent, exp_name, result_file, agent_name="Repairing")

    result_file.close()


def test_oracle_repair_force_5():
    _run_oracle_experiment("force_mag", "force_mag", injected_faults=[0.5])