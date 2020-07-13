import pytest
from agent.consistency.meta_model_repair import *
from agent.hydra_agent import *
from agent.planning.simple_planner import *
import worlds.science_birds as sb
from agent.consistency.consistency_estimator import check_obs_consistency, DEFAULT_DELTA_T
import matplotlib.pyplot

fh = logging.FileHandler("test_model_repair.log",mode='a')
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger = logging.getLogger("test_model_repair")
logger.setLevel(logging.DEBUG)
logger.addHandler(fh)

DATA_DIR = path.join(settings.ROOT_PATH, 'data')
TEST_DATA_DIR = path.join(DATA_DIR, 'science_birds', 'tests')
GRAVITY_FACTOR = "gravity_factor"


#################### System tests ########################
@pytest.fixture(scope="module")
def launch_science_birds_level_01():
    logger.info("Starting ScienceBirds")
    cmd = 'cp {}/data/science_birds/level-04.xml {}/00001.xml'.format(str(settings.ROOT_PATH), str(settings.SCIENCE_BIRDS_LEVELS_DIR))
    subprocess.run(cmd, shell=True)

    env = sb.ScienceBirds(None,launch=True,config='test_consistency_config.xml')
    yield env
    env.kill()
    logger.info("Ending ScienceBirds")

''' Adjusts game speed and ground truth frequency to obtain more observations'''
def _adjust_game_speed():
    settings.SB_SIM_SPEED = 1
    settings.SB_GT_FREQ = int(15 / settings.SB_SIM_SPEED)

''' Inject a fault to the agent's meta model '''
def _inject_fault_to_meta_model(meta_model : MetaModel, fluent_to_change = GRAVITY_FACTOR):
    meta_model.constant_numeric_fluents[fluent_to_change] = 6.0

''' A full system test: run SB with a bad meta model, observe results, fix meta model '''
@pytest.mark.skipif(settings.HEADLESS == True, reason="headless does not work in docker")
def test_repair_gravity_in_agent(launch_science_birds_level_01):
    # Constants
    save_obs = False # Set this to true to create a new observation file for test_repair_gravity_offline()
    plot_obs_vs_exp = False

    # Setup environment and agent
    _adjust_game_speed()
    env = launch_science_birds_level_01
    hydra = HydraAgent(env)

    # Inject fault and run the agent
    _inject_fault_to_meta_model(hydra.meta_model, GRAVITY_FACTOR)
    logger.info("Run agent, iteration 0")
    hydra.run_next_action()  # enough actions to play a level

    # Store observations
    if save_obs:
        observation = hydra.find_last_obs()
        obs_output_file = path.join(TEST_DATA_DIR, "obs_test_repair_gravity_in_agent.p")  # For debug
        pickle.dump(observation, open(obs_output_file, "wb"))  # For debug

    # Start repair loop
    consistency_checker = BirdLocationConsistencyEstimator()
    fluents_to_repair = [GRAVITY_FACTOR,]
    repair_deltas = [1.0, ]
    desired_precision = 20
    iteration = 0
    obs_with_rewards = 0
    meta_model = hydra.meta_model

    while iteration<6:
        observation = hydra.find_last_obs()

        # Store observation for debug
        if plot_obs_vs_exp:
            obs_output_file = path.join(DATA_DIR, "test_repair_gravity_in_agent_%d.p" % iteration) # For debug
            pickle.dump(observation, open(obs_output_file, "wb"))  # For debug
            matplotlib.interactive(True) # For debug
            fig = test_utils.plot_observation(observation) # For debug
            test_utils.plot_expected_trace_for_obs(hydra.meta_model, observation, ax=fig)
            matplotlib.pyplot.close()

        if observation.reward>0:
            logger.info("Reward ! (%.2f), iteration %d" % (observation.reward, iteration))
            obs_with_rewards = obs_with_rewards+1
            # No need to fix the model - we're winning! #TODO: This is an assumption: better to replace this with a good novelty detection mechanism
        else:
            consistency = check_obs_consistency(observation, meta_model, consistency_checker, delta_t=DEFAULT_DELTA_T)
            meta_model_repair = GreedyBestFirstSearchMetaModelRepair(fluents_to_repair, consistency_checker, repair_deltas,
                                                                     consistency_threshold=desired_precision)
            repair, _ = meta_model_repair.repair(meta_model, observation, delta_t=DEFAULT_DELTA_T)
            logger.info("Repair done (%s), iteration %d" % (repair,iteration))
            consistency_after = check_obs_consistency(observation, meta_model, consistency_checker, delta_t=DEFAULT_DELTA_T)
            assert consistency >= consistency_after # TODO: This actually may fail, because current model may be best, but we look for a different one

        # Run agent with repaired model
        iteration = iteration+1
        logger.info("Run agent, iteration %d" % iteration)
        hydra.run_next_action()  # enough actions to play a level

    assert obs_with_rewards>2 # Should at least win twice TODO: Ideally, this will check if we won both levels

''' Repair gravity based on an observed state
NOTE: If changed code that may affect the observations, rerun _inject_fault_and_run() with save_obs=True'''
def test_repair_gravity_offline():
    obs_output_file = path.join(TEST_DATA_DIR, "obs_test_repair_gravity_in_agent.p")
    observation = pickle.load(open(obs_output_file, "rb"))

    # Verify correct model is more consistent
    meta_model = MetaModel()
    good_consistency = check_obs_consistency(observation, meta_model, delta_t=DEFAULT_DELTA_T)

    _inject_fault_to_meta_model(meta_model)
    bad_consistency = check_obs_consistency(observation, meta_model, delta_t=DEFAULT_DELTA_T)

    assert bad_consistency > good_consistency

    # Repair model
    fluents_to_repair = [GRAVITY_FACTOR,]
    repair_deltas = [1.0,]
    desired_precision = 20
    consistency_checker = BirdLocationConsistencyEstimator()
    meta_model_repair = GreedyBestFirstSearchMetaModelRepair(fluents_to_repair, consistency_checker, repair_deltas,
                                                             consistency_threshold=desired_precision)
    meta_model_repair.repair(meta_model, observation, delta_t=DEFAULT_DELTA_T)
    repaired_consistency = check_obs_consistency(observation, meta_model, delta_t=DEFAULT_DELTA_T)
    assert bad_consistency > repaired_consistency