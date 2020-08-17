import matplotlib.pyplot as plt
import pytest
from agent.hydra_agent import *
from agent.planning.model_manipulator import ManipulateInitNumericFluent
from agent.planning.planner import *
from agent.perception.perception import *
import tests.test_utils as test_utils
from tests.test_utils import create_logger

logger = create_logger("test_consistency")

DATA_DIR = path.join(settings.ROOT_PATH, 'data')
TEST_DATA_DIR = path.join(DATA_DIR, 'science_birds', 'tests')
TEMP_TEST_DATA_DIR = path.join(TEST_DATA_DIR, 'temp')
PROBLEM_FILE = path.join(DATA_DIR, "sb_prob_l1.pddl")
DOMAIN_FILE = path.join(DATA_DIR, "sb_domain_l1.pddl")
PLAN_FILE = path.join(DATA_DIR, "docker_plan_trace_l1.txt")

PRECISION = 1
Y_BIRD_FLUENT = ('y_bird', 'redbird_0')
X_BIRD_FLUENT = ('x_bird', 'redbird_0')
DELTA_T = 0.05
GRAVITY_CHANGE = -50
GRAVITY = "gravity"

''' Helper function: loads plan, problem, and domain used to evaluate consistency checker'''
def _load_plan_problem_domain():
    (problem, domain) = test_utils.load_problem_and_domain(PROBLEM_FILE,DOMAIN_FILE)
    plan = Planner().extract_actions_from_plan_trace(PLAN_FILE)
    return (plan, problem, domain)

def _print_fluents_values(state_seq:list, fluent_names:list):
    headers = ",".join([str(fluent_name) for fluent_name in fluent_names])
    print("\n%s" % headers)
    for state in state_seq:
        line = ",".join(["%.2f" % state[fluent_name] for fluent_name in fluent_names])
        print("%s" % line)

''' Test the single numeric fluent consistency checker, for the case where the numeric fluent is not constant,
 and the outcome of the tests should be that the sequences are inconsistent. '''
def test_multiple_numeric_fluent_inconsistent():
    # Get expected timed state sequence according to the model (plan, problem, domain)
    (plan, problem, domain) = _load_plan_problem_domain()
    timed_state_seq = test_utils.simulate_plan_trace(plan, problem, domain, DEFAULT_DELTA_T)

    # Modify the model
    manipulator = ManipulateInitNumericFluent([GRAVITY],GRAVITY_CHANGE)
    manipulator.apply_change(domain, problem)

    # Get the new expected timed state sequence, according to the modified model
    simulation_trace = test_utils.simulate_plan_trace(plan, problem, domain, DEFAULT_DELTA_T)
    # Assert new timed state sequence is different from the original timed state sequence
    diff_list = diff_traces(timed_state_seq, simulation_trace)
    logger.info("\n".join(diff_list))
    assert len(diff_list)>0

    # Create a un-timed state sequence, to simulate how the SB observations would look like.
    modified_state_seq = [state for (state, t, _) in simulation_trace]

    # Assert the un-timed sequence created by the modified model is inconsistent with the timed sequence created by the original model
    consistency_checker = NumericFluentsConsistencyEstimator([X_BIRD_FLUENT, Y_BIRD_FLUENT],
                                                             obs_prefix=float('inf'),
                                                             discount_factor=1.0)
    consistent_score = consistency_checker.estimate_consistency(timed_state_seq, modified_state_seq)
    assert consistent_score > PRECISION


''' Test the single numeric fluent consistency checker, for the case where the numeric fluent is not constant,
 and the outcome of the tests should be that the sequences are inconsistent. '''
def test_all_bird_numeric_fluent_inconsistent():
    # Get expected timed state sequence according to the model (plan, problem, domain)
    (plan, problem, domain) = _load_plan_problem_domain()
    simulation_trace = test_utils.simulate_plan_trace(plan, problem, domain, DEFAULT_DELTA_T)

    consistency_checker = BirdLocationConsistencyEstimator(discount_factor=1.0, unique_prefix_size=float('inf'))
    # Assert the un-timed sequence created by the original model is consistent with the timed sequence created by the original model
    original_state_seq = [state for (state,t,_) in simulation_trace]
    consistency_score = consistency_checker.estimate_consistency(simulation_trace, original_state_seq)
    assert consistency_score < PRECISION


    # Modify the model
    manipulator = ManipulateInitNumericFluent([GRAVITY],GRAVITY_CHANGE)
    manipulator.apply_change(domain, problem)

    # Get the new expected timed state sequence, according to the modified model
    modified_simulation_trace =test_utils.simulate_plan_trace(plan, problem, domain, DEFAULT_DELTA_T)

    # Assert new timed state sequence is different from the original timed state sequence
    diff_list = diff_traces(simulation_trace, modified_simulation_trace)
    logger.info("\n".join(diff_list))
    assert len(diff_list)>0

    # Create a un-timed state sequence, to simulate how the SB observations would look like.
    modified_state_seq = [state for (state, t,_) in modified_simulation_trace]
    # Assert the un-timed sequence created by the modified model is inconsistent with the timed sequence created by the original model
    consistent_score = consistency_checker.estimate_consistency(simulation_trace, modified_state_seq)
    assert consistent_score > PRECISION

    consistent_score = consistency_checker.estimate_consistency(simulation_trace, modified_state_seq)
    assert consistent_score > PRECISION

#################### System Test ###############################
@pytest.fixture(scope="module")
def launch_science_birds():
    print("starting")
    env = sb.ScienceBirds(None,launch=True,config='test_consistency_config.xml')
    yield env
    print("teardown tests")
    env.kill()

''' Run Hydra, collect observations, check for consistency '''
# @pytest.mark.skipif(True, reason="Modified planner fails on basic levels")
def test_consistency_in_agent(launch_science_birds):
    settings.SB_SIM_SPEED = 5
    settings.SB_GT_FREQ = 1
    save_obs = True

    # Launch SB and play a level
    env = launch_science_birds
    hydra = HydraAgent(env)
    hydra.run_next_action()
    our_observation = hydra.find_last_obs()

    # Save the observation file for debug and for the offline test
    if save_obs:
        obs_output_file = path.join(TEST_DATA_DIR, "obs_test_consistency_in_agent.p")
    else:
        obs_output_file = path.join(TEMP_TEST_DATA_DIR, "obs_test_consistency_in_agent.p")

    pickle.dump(our_observation, open(obs_output_file, "wb")) #*** uncomment if needed for debugging ***

    plt.interactive(True)
    _, fig = plt.subplots()
    test_utils.plot_observation(our_observation, ax=fig)
    test_utils.plot_expected_trace_for_obs(hydra.meta_model, our_observation, ax=fig)

    # Check consistent with correct model
    consistency_estimator = BirdLocationConsistencyEstimator()
    good_consistency = check_obs_consistency(our_observation,hydra.meta_model, consistency_estimator)

    # Check consistent with incorrect model
    gravity_factor = hydra.meta_model.constant_numeric_fluents["gravity_factor"]
    bad_meta_model = MetaModel()
    bad_meta_model.constant_numeric_fluents["gravity_factor"] = gravity_factor/2
    bad_consistency = check_obs_consistency(our_observation, bad_meta_model, consistency_estimator)
    test_utils.plot_observation(our_observation, ax=fig)
    test_utils.plot_expected_trace_for_obs(bad_meta_model, our_observation, ax=fig)

    assert good_consistency < bad_consistency


''' Run Hydra, collect observations, check for consistency '''
# @pytest.mark.skipif(True, reason="Modified planner fails on basic levels")
def test_consistency_in_agent_offline():
    plot_exp_vs_obs = False

    obs_output_file = path.join(TEST_DATA_DIR, "obs_test_consistency_in_agent.p")
    our_observation = pickle.load(open(obs_output_file, "rb"))
    meta_model = MetaModel()

    # Uncomment for debug:
    # plt.interactive(True)
    # _, fig = plt.subplots()
    # fig = test_utils.plot_observation(our_observation, ax=fig)
    # fig = test_utils.plot_expected_trace_for_obs(meta_model, our_observation, ax=fig)

    # Check consistent with correct model
    consistency_estimator = BirdLocationConsistencyEstimator()
    good_consistency = check_obs_consistency(our_observation, meta_model, consistency_estimator, plot_obs_vs_exp=plot_exp_vs_obs)

    # Check consistent with incorrect model
    bad_meta_model = MetaModel()
    gravity_factor = meta_model.constant_numeric_fluents["gravity_factor"]
    bad_meta_model.constant_numeric_fluents["gravity_factor"] = gravity_factor/2
    bad_consistency = check_obs_consistency(our_observation, bad_meta_model, consistency_estimator, plot_obs_vs_exp=plot_exp_vs_obs)

    assert good_consistency < bad_consistency

''' Run Hydra, collect observations, check for consistency '''
# @pytest.mark.skipif(True, reason="Modified planner fails on basic levels")
def test_bad_shot_consistency(launch_science_birds):
    settings.SB_SIM_SPEED = 5
    settings.SB_GT_FREQ = 1
    plot_me = False
    save_obs = True

    env = launch_science_birds
    hydra = HydraAgent(env)
    angle = 75
    hydra.planner = PlannerStub(angle)
    hydra.run_next_action()
    meta_model = hydra.meta_model
    our_observation = hydra.find_last_obs()

    # Save the observation file for debug and for the offline test
    if save_obs:
        obs_output_file = path.join(TEST_DATA_DIR, "test_bad_shot_consistency_obs.p")
    else:
        obs_output_file = path.join(TEMP_TEST_DATA_DIR, "test_bad_shot_consistency_obs.p")
    pickle.dump(our_observation, open(obs_output_file, "wb"))  # *** uncomment if needed for debugging ***

    # For debug purposes
    # plt.interactive(True)
    # _, fig = plt.subplots()
    # fig = test_utils.plot_observation(our_observation, ax = fig)
    # fig = test_utils.plot_expected_trace_for_obs(hydra.meta_model, our_observation, ax=fig)

    # Check consistent with correct model
    consistency_estimator = BirdLocationConsistencyEstimator()
    consistency = check_obs_consistency(our_observation,meta_model, consistency_estimator, plot_obs_vs_exp=plot_me)

    # Check consistent with incorrect model
    good_gravity_factor = meta_model.constant_numeric_fluents["gravity_factor"]

    bad_meta_model = MetaModel()
    bad_meta_model.constant_numeric_fluents["gravity_factor"] = good_gravity_factor/2
    bad_consistency = check_obs_consistency(our_observation, bad_meta_model, consistency_estimator, plot_obs_vs_exp=plot_me)

    assert bad_consistency>consistency

def test_offline_bad_shot():
    plot_me = False
    meta_model = MetaModel()
    obs_output_file = path.join(TEST_DATA_DIR, "test_bad_shot_consistency_obs.p")
    our_observation = pickle.load(open(obs_output_file, "rb")) #*** uncomment if needed for debugging ***

    # For debug
    # plt.interactive(True)
    # _, fig = plt.subplots()
    # test_utils.plot_observation(our_observation, ax=fig)
    # fig = test_utils.plot_expected_trace_for_obs(meta_model, our_observation, ax=fig)

    # Check consistent with correct model
    consistency_estimator = BirdLocationConsistencyEstimator()
    consistency = check_obs_consistency(our_observation, meta_model, consistency_estimator, plot_obs_vs_exp=plot_me)

    # Check consistent with incorrect model
    good_gravity_factor = MetaModel().constant_numeric_fluents["gravity_factor"]

    bad_meta_model = MetaModel()
    bad_meta_model.constant_numeric_fluents["gravity_factor"] = good_gravity_factor/2
    bad_consistency = check_obs_consistency(our_observation, bad_meta_model, consistency_estimator, plot_obs_vs_exp=plot_me)

    assert bad_consistency>consistency