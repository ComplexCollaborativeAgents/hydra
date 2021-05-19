import pytest
from agent.repair.meta_model_repair import *
from agent.repair.sb_repair import BirdLocationConsistencyEstimator, ScienceBirdsConsistencyEstimator
from agent.hydra_agent import *
from agent.planning.simple_planner import *
import worlds.science_birds as sb
from agent.consistency.consistency_estimator import check_obs_consistency, DEFAULT_DELTA_T
import matplotlib.pyplot
from agent.repair.focused_repair import *

logger = test_utils.create_logger("test_model_repair")

DATA_DIR = path.join(settings.ROOT_PATH, 'data')
TEST_DATA_DIR = path.join(DATA_DIR, 'science_birds', 'tests')
GRAVITY_FACTOR = "gravity_factor"



''' Adjusts game speed and ground truth frequency to obtain more observations'''
def _adjust_game_speed():
    settings.SB_SIM_SPEED = 2
    settings.SB_GT_FREQ = int(30 / settings.SB_SIM_SPEED)

''' Inject a fault to the agent's meta model '''
def _inject_fault_to_meta_model(meta_model : ScienceBirdsMetaModel, fluent_to_change = GRAVITY_FACTOR):
    meta_model.constant_numeric_fluents[fluent_to_change] = 12

''' A full system test: run SB with a bad metnka model, observe results, fix meta model '''
@pytest.mark.skip("This test works, but is slow")
def test_repair_gravity_in_agent(save_obs=False,plot_obs_vs_exp=False):
    # Setup environment and agent
    env = sb.ScienceBirds(None, launch=True, config='test_consistency_config.xml')
    _adjust_game_speed()
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
    desired_precision = 2
    iteration = 0
    obs_with_rewards = 0
    meta_model = hydra.meta_model

    while iteration<3:
        observation = hydra.find_last_obs()

        # Store observation for debug
        obs_output_file = path.join(DATA_DIR, "test_repair_gravity_in_agent_obs_%d.p" % iteration)  # For debug
        pickle.dump(observation, open(obs_output_file, "wb"))  # For debug

        # Store observation for debug
        if plot_obs_vs_exp:
            matplotlib.interactive(True) # For debug
            fig = test_utils.plot_observation(observation) # For debug
            test_utils.plot_expected_trace_for_obs(hydra.meta_model, observation, ax=fig)
            matplotlib.pyplot.close()

        if observation.reward>0:
            logger.info("Reward ! (%.2f), iteration %d" % (observation.reward, iteration))
            obs_with_rewards = obs_with_rewards+1
            # No need to fix the model - we're winning! #TODO: This is an assumption: better to replace this with a good novelty detection mechanism
        else:
            consistency = check_obs_consistency(observation, meta_model, consistency_checker)
            meta_model_repair = GreedyBestFirstSearchMetaModelRepair(fluents_to_repair, consistency_checker, repair_deltas,
                                                                     consistency_threshold=desired_precision)
            repair, _ = meta_model_repair.repair(meta_model, observation, delta_t=DEFAULT_DELTA_T)
            logger.info("Repair done (%s), iteration %d" % (repair,iteration))
            consistency_after = check_obs_consistency(observation, meta_model, consistency_checker)
            assert consistency >= consistency_after # TODO: This actually may fail, because current model may be best, but we look for a different one

        # Run agent with repaired model
        iteration = iteration+1
        logger.info("Run agent, iteration %d" % iteration)
        hydra.run_next_action()  # enough actions to play a level

    assert obs_with_rewards>0 # Should at least win twice TODO: Ideally, this will check if we won both levels
    env.kill()

''' Repair gravity based on an observed state NOTE: If changed code that may affect the observations, rerun test_repair_gravity_in_agent() with save_obs=True'''
def test_repair_gravity_offline():
    # Repair model
    fluents_to_repair = [GRAVITY_FACTOR,]
    repair_deltas = [1.0,]
    desired_precision = 20
    consistency_estimator = ScienceBirdsConsistencyEstimator()
    meta_model_repair = GreedyBestFirstSearchMetaModelRepair(fluents_to_repair, consistency_estimator, repair_deltas,
                                                             consistency_threshold=desired_precision)

    _test_repair_gravity_offline(meta_model_repair)

''' Repair gravity based on an observed state NOTE: If changed code that may affect the observations, rerun test_repair_gravity_in_agent() with save_obs=True'''
def test_repair_gravity_offline_mma_repair():
    # Repair model
    fluents_to_repair = [GRAVITY_FACTOR,]
    repair_deltas = [1.0,]
    desired_precision = 20
    consistency_estimator = ScienceBirdsConsistencyEstimator()
    meta_model_repair = MmoBasedMetaModelRepair(fluents_to_repair, consistency_estimator, repair_deltas,
                                                             consistency_threshold=desired_precision)
    _test_repair_gravity_offline(meta_model_repair)

''' Repair gravity based on an observed state NOTE: If changed code that may affect the observations, rerun test_repair_gravity_in_agent() with save_obs=True'''
def test_repair_gravity_offline_focused_repair():
    # Repair model
    desired_precision = 20
    consistency_estimator = ScienceBirdsConsistencyEstimator()
    meta_model = ScienceBirdsMetaModel()

    meta_model_repair = FocusedMetaModelRepair(meta_model.repairable_constants, consistency_estimator, meta_model.repair_deltas,
                                                             consistency_threshold=desired_precision)
    _test_repair_gravity_offline(meta_model_repair)


''' Test repair gravity, using an observation file created by the test_repair_gravity_in_agent(True,...) method '''
def _test_repair_gravity_offline(meta_model_repair):
    obs_output_file = path.join(TEST_DATA_DIR, "obs_test_repair_gravity_in_agent.p")
    observation = pickle.load(open(obs_output_file, "rb"))
    consistency_estimator = ScienceBirdsConsistencyEstimator()
    # Verify correct model is more consistent
    meta_model = ScienceBirdsMetaModel()
    good_consistency = check_obs_consistency(observation, meta_model, consistency_estimator, plot_obs_vs_exp=True)
    _inject_fault_to_meta_model(meta_model)
    bad_consistency = check_obs_consistency(observation, meta_model, consistency_estimator, plot_obs_vs_exp=True)
    assert bad_consistency > good_consistency
    meta_model_repair.repair(meta_model, observation, delta_t=DEFAULT_DELTA_T)
    repaired_consistency = check_obs_consistency(observation, meta_model, consistency_estimator,
                                                 plot_obs_vs_exp=True)
    assert bad_consistency > repaired_consistency