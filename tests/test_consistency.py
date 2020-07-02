import pickle
import matplotlib.pyplot as plt
import pytest
import tests.test_utils as test_utils
from agent.hydra_agent import *
from agent.planning.model_manipulator import ManipulateInitNumericFluent
from agent.planning.planner import *
from tests.test_utils import PlannerStub
import settings
from agent.perception.perception import *

fh = logging.FileHandler("hydra.log",mode='w')
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger = logging.getLogger("test_consistency")
logger.setLevel(logging.INFO)
logger.addHandler(fh)

PRECISION = 1
DATA_DIR = path.join(settings.ROOT_PATH, 'data')
TRACE_DIR = path.join(DATA_DIR, 'science_birds', 'serialized_levels', 'level-01')

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
    plan = test_utils.load_plan(PLAN_FILE, problem,domain)
    return (plan, problem, domain)

def _print_fluents_values(state_seq:list, fluent_names:list):
    headers = ",".join([str(fluent_name) for fluent_name in fluent_names])
    print("\n%s" % headers)
    for state in state_seq:
        line = ",".join(["%.2f" % state[fluent_name] for fluent_name in fluent_names])
        print("%s" % line)


''' Test the single numeric fluent consistency checker, for the case where the numeric fluent is constant,
 and the outcome of the tests should be that the sequences are consistent. '''
def test_single_constant_numeric_fluent_consistent():
    # Get expected timed state sequence according to the model (plan, problem, domain)
    (plan, problem, domain) = _load_plan_problem_domain()
    timed_state_seq = test_utils.simulate_plan_trace(plan, problem, domain, DELTA_T)
    assert len(timed_state_seq)>0 # Sanity check

    # Create a un-timed state sequence, to simulate how the SB observations would look like.
    state_seq = [state for (state, t,_) in timed_state_seq]

    # Check that the un-timed state sequence is consistent with itself
    consistency_checker = SingleNumericFluentConsistencyEstimator((GRAVITY))
    assert consistency_checker.estimate_consistency(timed_state_seq, state_seq)==0

    # Check that the un-timed state sequence is consistent with a subset of itself
    partial_state_seq = state_seq[:(len(state_seq)-5)]
    assert consistency_checker.estimate_consistency(timed_state_seq, partial_state_seq) == 0

    partial_state_seq = [state_seq[i] for i in range(0,len(state_seq),2)] # Get every other state
    assert consistency_checker.estimate_consistency(timed_state_seq, partial_state_seq) == 0

    partial_state_seq = [state_seq[i] for i in range(0,len(state_seq),4)] # Get every forth state
    assert consistency_checker.estimate_consistency(timed_state_seq, partial_state_seq) == 0


''' Test the single numeric fluent consistency checker, for the case where the numeric fluent is constant,
 and the outcome of the tests should be that the sequences are inconsistent. '''
def test_single_constant_numeric_fluent_inconsistent():
    # Get expected timed state sequence according to the model (plan, problem, domain)
    (plan, problem, domain) = _load_plan_problem_domain()
    simulation_trace = test_utils.simulate_plan_trace(plan, problem, domain, DELTA_T)

    # Modify the model
    manipulator = ManipulateInitNumericFluent([GRAVITY],GRAVITY_CHANGE)
    manipulator.apply_change(domain, problem)

    # Get the new expected timed state sequence, according to the modified model
    modified_simulation_trace = test_utils.simulate_plan_trace(plan, problem, domain, DELTA_T)
    simulation_trace = simulation_trace[:len(modified_simulation_trace)] # Trim to compare

    # Compute the diff between the timed state sequence and the not-timed state sequence. There should be none.
    diff_list = diff_traces(simulation_trace, modified_simulation_trace)
    logger.info("\n".join(diff_list))
    assert len(diff_list)>0 # Assert expected timed state sequence has changed due to the change in the model

    # Create a un-timed state sequence, to simulate how the SB observations would look like.
    modified_state_seq = [state for (state, t,_) in modified_simulation_trace]

    # Assert the un-timed sequence created by the modified model is inconsistent with the timed sequence created by the original model
    consistency_checker = SingleNumericFluentConsistencyEstimator(X_BIRD_FLUENT)
    consistent_score = consistency_checker.estimate_consistency(simulation_trace, modified_state_seq)
    assert consistent_score > 0.0


''' Test the single numeric fluent consistency checker, for the case where the numeric fluent is not constant,
 and the outcome of the tests should be that the sequences are consistent. '''
def test_single_constant_numeric_fluent_consistent():
    # Get expected timed and un-timed state sequence according to the model (plan, problem, domain)
    (plan, problem, domain) = _load_plan_problem_domain()
    simulation_trace = test_utils.simulate_plan_trace(plan, problem, domain, DELTA_T)

    state_seq = [state for (state, t,_) in simulation_trace]

    # Check that the un-timed sequence is consistent with the timed sequence, w.r.t. the X_REDBIRD fluent
    consistency_checker = SingleNumericFluentConsistencyEstimator(Y_BIRD_FLUENT)
    assert consistency_checker.estimate_consistency(simulation_trace, state_seq) < PRECISION

    # Check that subsets of the un-timed sequence are consistent with the timed sequence, w.r.t. the X_REDBIRD fluent
    partial_state_seq = state_seq[:(len(state_seq)-5)]
    assert consistency_checker.estimate_consistency(simulation_trace, partial_state_seq) < PRECISION

    partial_state_seq = [state_seq[i] for i in range(0,len(state_seq),2)] # Get every other state
    assert consistency_checker.estimate_consistency(simulation_trace, partial_state_seq) < PRECISION

    partial_state_seq = [state_seq[i] for i in range(0,len(state_seq),4)] # Get every forth state
    assert consistency_checker.estimate_consistency(simulation_trace, partial_state_seq) < PRECISION


''' Test the single numeric fluent consistency checker, for the case where the numeric fluent is not constant,
 and the outcome of the tests should be that the sequences are inconsistent. '''
def test_single_numeric_fluent_inconsistent():
    # Get expected timed state sequence according to the model (plan, problem, domain)
    (plan, problem, domain) = _load_plan_problem_domain()
    timed_state_seq = test_utils.simulate_plan_trace(plan, problem, domain, DELTA_T)

    # Modify the model
    manipulator = ManipulateInitNumericFluent([GRAVITY],GRAVITY_CHANGE)
    manipulator.apply_change(domain, problem)

    # Get the new expected timed state sequence, according to the modified model
    modified_timed_state_seq = test_utils.simulate_plan_trace(plan, problem, domain, DELTA_T)

    # Assert new timed state sequence is different from the original timed state sequence
    diff_list = diff_traces(timed_state_seq, modified_timed_state_seq)
    logger.info("\n".join(diff_list))
    assert len(diff_list)>0

    # Create a un-timed state sequence, to simulate how the SB observations would look like.
    modified_state_seq = [state for (state, t,_) in modified_timed_state_seq]

    # Assert the un-timed sequence created by the modified model is inconsistent with the timed sequence created by the original model
    consistency_checker = SingleNumericFluentConsistencyEstimator(Y_BIRD_FLUENT)
    consistent_score = consistency_checker.estimate_consistency(timed_state_seq, modified_state_seq)
    assert consistent_score > PRECISION


''' Test the single numeric fluent consistency checker, for the case where the numeric fluent is not constant,
 and the outcome of the tests should be that the sequences are inconsistent. '''
def test_multiple_numeric_fluent_inconsistent():
    # Get expected timed state sequence according to the model (plan, problem, domain)
    (plan, problem, domain) = _load_plan_problem_domain()
    timed_state_seq = test_utils.simulate_plan_trace(plan, problem, domain, DELTA_T)

    # Modify the model
    manipulator = ManipulateInitNumericFluent([GRAVITY],GRAVITY_CHANGE)
    manipulator.apply_change(domain, problem)

    # Get the new expected timed state sequence, according to the modified model
    simulation_trace = test_utils.simulate_plan_trace(plan, problem, domain, DELTA_T)
    # Assert new timed state sequence is different from the original timed state sequence
    diff_list = diff_traces(timed_state_seq, simulation_trace)
    logger.info("\n".join(diff_list))
    assert len(diff_list)>0

    # Create a un-timed state sequence, to simulate how the SB observations would look like.
    modified_state_seq = [state for (state, t, _) in simulation_trace]

    # Assert the un-timed sequence created by the modified model is inconsistent with the timed sequence created by the original model
    consistency_checker = NumericFluentsConsistencyEstimator([X_BIRD_FLUENT, Y_BIRD_FLUENT],
                                                             unique_prefix_size=float('inf'),
                                                             discount_factor=1.0)
    consistent_score = consistency_checker.estimate_consistency(timed_state_seq, modified_state_seq)
    assert consistent_score > PRECISION


''' Test the single numeric fluent consistency checker, for the case where the numeric fluent is not constant,
 and the outcome of the tests should be that the sequences are inconsistent. '''
def test_all_bird_numeric_fluent_inconsistent():
    # Get expected timed state sequence according to the model (plan, problem, domain)
    (plan, problem, domain) = _load_plan_problem_domain()
    simulation_trace = test_utils.simulate_plan_trace(plan, problem, domain, DELTA_T)

    consistency_checker = BirdLocationConsistencyEstimator(discount_factor=1.0, unique_prefix_size=float('inf'))
    # Assert the un-timed sequence created by the original model is consistent with the timed sequence created by the original model
    original_state_seq = [state for (state,t,_) in simulation_trace]
    consistency_score = consistency_checker.estimate_consistency(simulation_trace, original_state_seq)
    assert consistency_score < PRECISION


    # Modify the model
    manipulator = ManipulateInitNumericFluent([GRAVITY],GRAVITY_CHANGE)
    manipulator.apply_change(domain, problem)

    # Get the new expected timed state sequence, according to the modified model
    modified_simulation_trace =test_utils.simulate_plan_trace(plan, problem, domain, DELTA_T)

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
def test_consistency_in_agent_with_dummy_planner(launch_science_birds):
    settings.SB_SIM_SPEED = 1
    settings.SB_GT_FREQ = int(15 / settings.SB_SIM_SPEED)

    env = launch_science_birds
    hydra = HydraAgent(env)

    # Run agent with dummy action and collect observation
    raw_timed_action = ["pa-twang redbird_0", 65.5]
    plan = [raw_timed_action]
    hydra.planner = PlannerStub(plan, hydra.meta_model)
    hydra.main_loop(max_actions=3)  # enough actions to play the first level
    our_observation =  hydra.observations[-1]
    # pickle.dump(our_observation,open("twang_65.p", "wb")) #*** uncomment if needed for debugging ***
    observed_seq = our_observation.get_trace(hydra.meta_model)

    plt.interactive(True)
    test_utils.plot_observation(our_observation)

    # Inject fault in meta model and compute expected trace
    ANGLE_RATE = 'angle_rate'
    meta_model = MetaModel()
    meta_model.constant_numeric_fluents[ANGLE_RATE] = 40
    problem = meta_model.create_pddl_problem(our_observation.state)
    domain = meta_model.create_pddl_domain(our_observation.state)
    # PddlDomainExporter().to_file(domain, domain_output_file) *** uncomment if needed for debugging ***
    # PddlProblemExporter().to_file(problem, problem_output_file) *** uncomment if needed for debugging ***
    grounded_domain = PddlPlusGrounder().ground_domain(domain,problem)

    pddl_plan = PddlPlusPlan()
    pddl_plan.add_raw_actions(plan, grounded_domain)
    expected_timed_state_seq = test_utils.simulate_plan_trace(pddl_plan, problem, domain, DELTA_T)
    pddl_state = meta_model.create_pddl_state(our_observation.state)
    test_utils.plot_state_sequence([timed_state[0] for timed_state in expected_timed_state_seq], pddl_state)

    consistency_estimator = BirdLocationConsistencyEstimator()
    consistency_value_for_faulty_model = consistency_estimator.estimate_consistency(expected_timed_state_seq, observed_seq)

    assert consistency_value_for_faulty_model>PRECISION

    # Repair fault, and check consistency value
    meta_model = MetaModel()
    meta_model.constant_numeric_fluents[ANGLE_RATE] = settings.SB_SIM_SPEED *20/30 # Allowing me to play with the simulation speed

    problem = meta_model.create_pddl_problem(our_observation.state)
    domain = meta_model.create_pddl_domain(our_observation.state)
    expected_timed_state_seq = test_utils.simulate_plan_trace(pddl_plan, problem, domain, DELTA_T)
    pddl_state = meta_model.create_pddl_state(our_observation.state)
    test_utils.plot_state_sequence([timed_state[0] for timed_state in expected_timed_state_seq], pddl_state)

    consistency_value_for_healthy_model = consistency_estimator.estimate_consistency(expected_timed_state_seq, observed_seq)
    assert consistency_value_for_faulty_model>consistency_value_for_healthy_model