import settings
from agent.repairing_hydra_agent import RepairingHydraAgent
import pytest
from agent.planning.pddl_meta_model import *
import subprocess
import worlds.science_birds as sb
import pickle
import tests.test_utils as test_utils
import os.path as path

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
def _inject_fault_to_meta_model(meta_model : MetaModel, fluent_to_change = GRAVITY_FACTOR):
    meta_model.constant_numeric_fluents[fluent_to_change] = 6.0


''' A full system test: run SB with a bad meta model, observe results, fix meta model '''
@pytest.mark.skipif(settings.HEADLESS == True, reason="headless does not work in docker")
def test_repair_gravity_in_agent(launch_science_birds):
    # Setup environment and agent
    save_obs = True
    plot_exp_vs_obs = True

    settings.SB_SIM_SPEED=5
    settings.SB_GT_FREQ =1

    env = launch_science_birds
    hydra = RepairingHydraAgent(env)

    # Inject fault and run the agent
    _inject_fault_to_meta_model(hydra.meta_model, GRAVITY_FACTOR)

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