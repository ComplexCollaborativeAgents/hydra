import settings
from agent.repairing_hydra_agent import RepairingHydraAgent
from agent.hydra_agent import HydraAgent
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

# Setup environment and agent
save_obs = True
plot_exp_vs_obs = False
settings.SB_SIM_SPEED = 5
settings.SB_GT_FREQ = 1

logger = test_utils.create_logger("test_repairing_experiments")

@pytest.fixture(scope="module")
def launch_science_birds_with_all_levels():
    logger.info("Starting ScienceBirds")
    env = sb.ScienceBirds(None,launch=True,config='default_ANU_config.xml')
    yield env
    env.kill()
    logger.info("Ending ScienceBirds")

''' Run an experiment'''
def _run_experiment(hydra, experiment_name, max_iterations = 10):
    # Inject fault and run the agenth
    # _inject_fault_to_meta_model(hydra.meta_model, GRAVITY_FACTOR)
    try:
        results_file = open(path.join(TEST_DATA_DIR, "%s.csv" % experiment_name), "w")
        results_file.write("Iteration\t Reward\t PlanningTime\t CummulativePlanningTime\t Gravity\n")

        iteration = 0
        obs_with_rewards = 0
        while iteration < max_iterations:
            hydra.run_next_action()
            observation = hydra.find_last_obs()

            results_file.write("%d\t %.2f\t %.2f\t %.2f\t %.2f\n" % (iteration,
                                                                     observation.reward,
                                                                     hydra.overall_plan_time,
                                                                     hydra.cumulative_plan_time,
                                                                     hydra.meta_model.constant_numeric_fluents[GRAVITY_FACTOR]))
            results_file.flush()

            # Store observation for debug
            if save_obs:
                obs_file = path.join(TEST_DATA_DIR, "%s-%d.obs" % (experiment_name, iteration))  # For debug
                meta_model_file = path.join(TEST_DATA_DIR, "%s-%d.mm" % (experiment_name, iteration))  # For debug
                pickle.dump(observation, open(obs_file, "wb"))  # For debug
                pickle.dump(hydra.meta_model, open(meta_model_file, "wb"))
            if plot_exp_vs_obs:
                test_utils.plot_expected_vs_observed(hydra.meta_model, observation)

            if observation.reward > 0:
                logger.info("Reward ! (%.2f), iteration %d" % (observation.reward, iteration))
                obs_with_rewards = obs_with_rewards + 1
            iteration = iteration + 1
        assert obs_with_rewards > 0
    finally:
        results_file.close()

@pytest.mark.skip("Have not migrated to 0.3.6 yet")
def test_set_of_levels_no_repair(launch_science_birds_with_all_levels):
    env = launch_science_birds_with_all_levels
    hydra = HydraAgent(env)
    max_iterations = 10
    _run_experiment(hydra, "no_repair-%d" % max_iterations, max_iterations=max_iterations)

@pytest.mark.skip("Have not migrated to 0.3.6 yet")
def test_set_of_levels_repair_no_fault(launch_science_birds_with_all_levels):
    env = launch_science_birds_with_all_levels
    hydra = RepairingHydraAgent(env)
    max_iterations = 10
    _run_experiment(hydra, "with_repair-%d" % max_iterations, max_iterations=max_iterations)


''' Inject a fault to the agent's meta model '''
def _inject_fault_to_meta_model(meta_model : MetaModel, fluent_to_change = GRAVITY_FACTOR):
    meta_model.constant_numeric_fluents[fluent_to_change] = 6.0

@pytest.mark.skip("Have not migrated to 0.3.6 yet")
def test_set_of_levels_repair_with_fault(launch_science_birds_with_all_levels):
    env = launch_science_birds_with_all_levels
    hydra = RepairingHydraAgent(env)
    _inject_fault_to_meta_model(hydra.meta_model, GRAVITY_FACTOR)
    max_iterations = 10
    _run_experiment(hydra, "with_repair_bad_gravity-%d" % max_iterations, max_iterations=max_iterations)

@pytest.mark.skip("Have not migrated to 0.3.6 yet")
def test_set_of_levels_no_repair_with_fault(launch_science_birds_with_all_levels):
    env = launch_science_birds_with_all_levels
    hydra = HydraAgent(env)
    _inject_fault_to_meta_model(hydra.meta_model, GRAVITY_FACTOR)
    max_iterations = 10
    _run_experiment(hydra, "no_repair_bad_gravity-%d" % max_iterations, max_iterations=max_iterations)
