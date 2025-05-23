import pytest
from agent.repair.meta_model_repair import *
from agent.gym_hydra_agent import GymHydraAgent
from agent.planning.simple_sb_planner import *
from agent.consistency.consistency_estimator import check_obs_consistency, DEFAULT_DELTA_T
from agent.repair.cartpole_repair import CartpoleConsistencyEstimator
import gym
import tests.test_utils as test_utils
logger = test_utils.create_logger("test_model_repair")

DATA_DIR = path.join(settings.ROOT_PATH, 'data')
TEST_DATA_DIR = path.join(DATA_DIR, 'science_birds', 'tests')
GRAVITY = "gravity"


@pytest.fixture(scope="module")
def launch_cartpole_sample_level():
    logger.info("Starting CartPole")
    env = gym.make("CartPole-v1")
    yield env
    # env.kill()
    logger.info("Ending CartPole")


''' Inject a fault to the agent's meta model '''
def _inject_fault_to_meta_model(meta_model : CartPoleMetaModel, fluent_to_change = GRAVITY):
    meta_model.constant_numeric_fluents[fluent_to_change] = 4.9

''' A full system test: run SB with a bad metnka model, observe results, fix meta model '''

# @pytest.mark.skip()
def test_repair_gravity_in_cartpole_agent(launch_cartpole_sample_level):
    # Constants
    save_obs = True # Set this to true to create a new observation file for test_repair_gravity_offline()
    plot_obs_vs_exp = False

    # Setup environment and agent
    env = launch_cartpole_sample_level
    cartpole_hydra = GymHydraAgent(env)


    # Inject fault and run the agent
    _inject_fault_to_meta_model(cartpole_hydra.meta_model, GRAVITY)
    logger.info("Run agent, iteration 0")
    # hydra.run_next_action()  # enough actions to play a level

    cartpole_hydra.run()

    # Store observations
    if save_obs:
        observation = cartpole_hydra.find_last_obs()
        obs_output_file = path.join(TEST_DATA_DIR, "obs_test_repair_gravity_in_cartpole_agent.p")  # For debug
        pickle.dump(observation, open(obs_output_file, "wb"))  # For debug

    # Start repair loop
    consistency_checker = CartpoleConsistencyEstimator()
    fluents_to_repair = [GRAVITY,]
    repair_deltas = [1.0, ]
    desired_precision = 0.01
    iteration = 0
    obs_with_rewards = 0
    meta_model = cartpole_hydra.meta_model

    while iteration<3:
        observation = cartpole_hydra.find_last_obs()

        # Store observation for debug
        obs_output_file = path.join(DATA_DIR, "test_repair_gravity_in_cartpole_agent_obs_%d.p" % iteration)  # For debug
        pickle.dump(observation, open(obs_output_file, "wb"))  # For debug

        # Store observation for debug
        # if plot_obs_vs_exp:
        #     matplotlib.interactive(True) # For debug
        #     fig = test_utils.plot_observation(observation) # For debug
        #     test_utils.plot_expected_trace_for_obs(cartpole_hydra.meta_model, observation, ax=fig)
        #     matplotlib.pyplot.close()

        logger.info("Reward ! (%.2f), iteration %d" % (sum(observation.rewards), iteration))
        logger.info("Actions ! (%.2f), iteration %d" % (len(observation.actions), iteration))
        logger.info("States ! (%.2f), iteration %d" % (len(observation.states), iteration))

        if sum(observation.rewards)>195:
            logger.info("Finished with correct reward...")
            obs_with_rewards = obs_with_rewards+1
            break
            # No need to fix the model - we're winning! #TODO: This is an assumption: better to replace this with a good novelty detection mechanism
        else:
            consistency = check_obs_consistency(observation, meta_model, consistency_checker)
            meta_model_repair = GreedyBestFirstSearchConstantFluentMetaModelRepair(fluents_to_repair, consistency_checker, repair_deltas,
                                                                                   consistency_threshold=desired_precision)
            repair, _ = meta_model_repair.repair(meta_model, observation, delta_t=DEFAULT_DELTA_T)
            logger.info("Repair done (%s), iteration %d" % (repair,iteration))
            consistency_after = check_obs_consistency(observation, meta_model, consistency_checker)
            assert consistency >= consistency_after # TODO: This actually may fail, because current model may be best, but we look for a different one

        # Run agent with repaired model
        iteration = iteration+1
        logger.info("Run agent, iteration %d" % iteration)

        # cartpole_hydra.observation = cartpole_hydra.reset_with_seed()
        cartpole_hydra.observation = cartpole_hydra.env.reset()
        cartpole_hydra.run()  # enough actions to play a level

    assert obs_with_rewards>0 # Should at least win twice TODO: Ideally, this will check if we won both levels

''' Repair gravity based on an observed state
NOTE: If changed code that may affect the observations, rerun _inject_fault_and_run() with save_obs=True'''
@pytest.mark.skip()
def test_repair_gravity_offline():
    obs_output_file = path.join(TEST_DATA_DIR, "obs_test_repair_gravity_in_cartpole_agent.p")
    observation = pickle.load(open(obs_output_file, "rb"))

    # Verify correct model is more consistent
    meta_model = CartPoleMetaModel()
    consistency_checker = CartpoleConsistencyEstimator()
    good_consistency = check_obs_consistency(observation, meta_model, consistency_checker)

    _inject_fault_to_meta_model(meta_model, GRAVITY)

    bad_consistency = check_obs_consistency(observation, meta_model, consistency_checker)

    assert bad_consistency > good_consistency

    # Repair model
    fluents_to_repair = [GRAVITY,]
    repair_deltas = [1.0,]
    desired_precision = 0.01
    meta_model_repair = GreedyBestFirstSearchConstantFluentMetaModelRepair(fluents_to_repair, consistency_checker, repair_deltas,
                                                                           consistency_threshold=desired_precision)

    meta_model_repair.repair(meta_model, observation, delta_t=DEFAULT_DELTA_T)
    repaired_consistency = check_obs_consistency(observation, meta_model, consistency_checker)
    assert bad_consistency > repaired_consistency



''' Inject a fault to the agent's meta model '''
def _inject_fault_to_gym(env):
    env.env.gravity = 4.9

# @pytest.mark.skip()
def test_repair_bad_gravity_in_gym(launch_cartpole_sample_level):
    save_obs = True # Set this to true to create a new observation file for an offline test

    # Setup environment and agent
    env = launch_cartpole_sample_level
    cartpole_hydra = GymHydraAgent(env)


    # Inject fault and run the agent
    _inject_fault_to_gym(env)
    logger.info("Run agent, iteration 0")

    cartpole_hydra.run()
    # Store observations
    if save_obs:
        observation = cartpole_hydra.find_last_obs()
        obs_output_file = path.join(TEST_DATA_DIR, "obs_test_repair_bad_gravity_in_gym.p")  # For debug
        pickle.dump(observation, open(obs_output_file, "wb"))  # For debug

    # Start repair loop
    consistency_checker = CartpoleConsistencyEstimator()


    fluents_to_repair = [GRAVITY,]
    repair_deltas = [1.0, ]
    desired_precision = 0.01
    iteration = 0
    obs_with_rewards = 0
    meta_model = cartpole_hydra.meta_model
    while iteration<3:
        observation = cartpole_hydra.find_last_obs()

        # Store observation for debug
        obs_output_file = path.join(DATA_DIR, "obs_test_repair_bad_gravity_in_gym_%d.p" % iteration)  # For debug
        pickle.dump(observation, open(obs_output_file, "wb"))  # For debug

        logger.info("Reward ! (%.2f), iteration %d" % (sum(observation.rewards), iteration))
        logger.info("Actions ! (%.2f), iteration %d" % (len(observation.actions), iteration))
        logger.info("States ! (%.2f), iteration %d" % (len(observation.states), iteration))

        if sum(observation.rewards)>195:
            logger.info("Finished with correct reward...")
            obs_with_rewards = obs_with_rewards+1
            break
            # No need to fix the model - we're winning! #TODO: This is an assumption: better to replace this with a good novelty detection mechanism
        else:
            consistency = check_obs_consistency(observation, meta_model, consistency_checker)
            meta_model_repair = GreedyBestFirstSearchConstantFluentMetaModelRepair(fluents_to_repair, consistency_checker, repair_deltas,
                                                                                   consistency_threshold=desired_precision)
            repair, _ = meta_model_repair.repair(meta_model, observation, delta_t=meta_model.delta_t)
            logger.info("Repair done (%s), iteration %d" % (repair,iteration))
            consistency_after = check_obs_consistency(observation, meta_model, consistency_checker)
            assert consistency >= consistency_after # TODO: This actually may fail, because current model may be best, but we look for a different one

        # Run agent with repaired model
        iteration = iteration+1
        logger.info("Run agent, iteration %d" % iteration)

        cartpole_hydra.observation = cartpole_hydra.env.reset()
        cartpole_hydra.run()  # enough actions to play a level

    assert obs_with_rewards>0 # Should at least win twice TODO: Ideally, this will check if we won both levels
