import matplotlib.pyplot as plt
import pytest
from agent.hydra_agent import *
from agent.planning.model_manipulator import ManipulateInitNumericFluent
from agent.planning.planner import *
from agent.perception.perception import *

fh = logging.FileHandler("hydra.log",mode='w')
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger = logging.getLogger("test_consistency")
logger.setLevel(logging.INFO)
logger.addHandler(fh)

PRECISION = 1
DATA_DIR = path.join(settings.ROOT_PATH, 'data')
TEMP_DIR = path.join(settings.ROOT_PATH, 'data') # todo: not ideal that this is just the DATA
TEST_DATA_DIR = path.join(settings.ROOT_PATH, 'data', 'science_birds', 'tests')

Y_BIRD_FLUENT = ('y_bird', 'redbird_0')
X_BIRD_FLUENT = ('x_bird', 'redbird_0')
DELTA_T = 0.05
GRAVITY_CHANGE = -50
GRAVITY = "gravity"

PROBLEM_FILE = path.join(DATA_DIR, "sb_prob_l1.pddl")
DOMAIN_FILE = path.join(DATA_DIR, "sb_domain_l1.pddl")
PLAN_FILE = path.join(DATA_DIR, "docker_plan_trace_l1.txt")

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
    logging.info("\n".join(diff_list))
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
    logging.info("\n".join(diff_list))
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
    save_obs = True

    env = launch_science_birds
    hydra = HydraAgent(env)
    hydra.run_next_action()
    our_observation = hydra.find_last_obs()

    if save_obs:
        obs_output_file = path.join(TEST_DATA_DIR, "obs_test_consistency_in_agent.p")
    else:
        obs_output_file = path.join(TEMP_DIR, "obs_test_consistency_in_agent.p")  # For debug

    pickle.dump(our_observation, open(obs_output_file, "wb")) #*** uncomment if needed for debugging ***

    plt.interactive(True)
    fig = test_utils.plot_observation(our_observation)
    fig = test_utils.plot_expected_trace_for_obs(hydra.meta_model, our_observation, ax=fig)

    # Check consistent with correct model
    consistency_estimator = BirdLocationConsistencyEstimator()
    good_consistency = check_obs_consistency(our_observation,hydra.meta_model, consistency_estimator)

    # Check consistent with incorrect model
    gravity_factor = hydra.meta_model.constant_numeric_fluents["gravity_factor"]
    bad_meta_model = MetaModel()
    bad_meta_model.constant_numeric_fluents["gravity_factor"] = gravity_factor/2
    bad_consistency = check_obs_consistency(our_observation, bad_meta_model, consistency_estimator)

    assert good_consistency < bad_consistency


''' Run Hydra, collect observations, check for consistency '''
# @pytest.mark.skipif(True, reason="Modified planner fails on basic levels")
def test_consistency_in_agent_offline():
    plot_obs_vs_exp = False
    obs_output_file = path.join(TEST_DATA_DIR, "obs_test_consistency_in_agent.p")
    our_observation = pickle.load(open(obs_output_file, "rb"))
    meta_model = MetaModel()

    if plot_obs_vs_exp:
        plt.interactive(True)
        fig = test_utils.plot_observation(our_observation)
        fig = test_utils.plot_expected_trace_for_obs(meta_model, our_observation, ax=fig)

    # Check consistent with correct model
    consistency_estimator = BirdLocationConsistencyEstimator()
    good_consistency = check_obs_consistency(our_observation, meta_model, consistency_estimator)

    # Check consistent with incorrect model
    gravity_factor = meta_model.constant_numeric_fluents["gravity_factor"]
    bad_meta_model = MetaModel()
    bad_meta_model.constant_numeric_fluents["gravity_factor"] = gravity_factor/2
    bad_consistency = check_obs_consistency(our_observation, bad_meta_model, consistency_estimator)

    assert good_consistency < bad_consistency
