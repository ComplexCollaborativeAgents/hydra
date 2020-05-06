import pytest
import worlds.science_birds as sb
from agent.perception.perception import Perception
from agent.consistency.consistency_checker import *
from agent.consistency.pddl_plus_simulator import *
from agent.planning.planner import *
from agent.planning.pddlplus_parser import *
from agent.planning.model_manipulator import ManipulateInitNumericFluent
import logging

fh = logging.FileHandler("hydra.log",mode='w')
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger = logging.getLogger("test_consistency")
logger.setLevel(logging.INFO)
logger.addHandler(fh)

PRECISION = 1
DATA_DIR = path.join(settings.ROOT_PATH, 'data')
TRACE_DIR = path.join(DATA_DIR, 'science_birds', 'serialized_levels', 'level-01')

''' Helper function: simulate the given plan, on the given problem and domain.  '''
def get_timed_state_seq(domain: PddlPlusDomain, problem:PddlPlusProblem, plan: PddlPlusPlan, delta_t:float):
    simulator = PddlPlusSimulator()
    (_, _, trace) =  simulator.simulate(plan, problem, domain, delta_t)
    return trace

''' Return plan, problem, and domain used to evaluate consistency checker'''
@pytest.fixture(scope="module")
def get_plan_problem_domain():
    problem_file = path.join(DATA_DIR, "sb_prob_l1.pddl")
    problem_parser = PddlProblemParser()
    pddl_problem = problem_parser.parse_pddl_problem(problem_file)
    assert pddl_problem is not None, "PDDL+ problem object not parsed"  # Sanity check, parser works

    domain_file = path.join(DATA_DIR, "sb_domain_l1.pddl")
    domain_parser = PddlDomainParser()
    pddl_domain = domain_parser.parse_pddl_domain(domain_file)
    assert pddl_domain is not None, "PDDL+ domain object not parsed"  # Sanity check, parser works

    planner = Planner()
    grounded_domain = PddlPlusGrounder().ground_domain(pddl_domain, pddl_problem)  # Needed to identify plan action
    plan_trace_file = path.join(DATA_DIR, "docker_plan_trace_l1.txt")
    pddl_plan = planner.extract_plan_from_plan_trace(plan_trace_file, grounded_domain)
    return (pddl_plan, pddl_problem, pddl_domain)

''' Test the single numeric fluent consistency checker, for the case where the numeric fluent is constant,
 and the outcome of the tests should be that the sequences are consistent. '''
def test_single_constant_numeric_fluent_consistent(get_plan_problem_domain):
    DELTA_T = 0.05
    GRAVITY = ("gravity")

    # Get expected timed state sequence according to the model (plan, problem, domain)
    (pddl_plan, pddl_problem, pddl_domain) = get_plan_problem_domain
    timed_state_seq = get_timed_state_seq(pddl_domain, pddl_problem, pddl_plan, DELTA_T)
    assert len(timed_state_seq)>0 # Sanity check

    # Create a un-timed state sequence, to simulate how the SB observations would look like.
    state_seq = [state for (state, t) in timed_state_seq]

    # Check that the un-timed state sequence is consistent with itself
    consistency_checker = SingleNumericFluentConsistencyChecker(GRAVITY)
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
def test_single_constant_numeric_fluent_inconsistent(get_plan_problem_domain):
    GRAVITY = "gravity"
    DELTA_T = 0.05
    GRAVITY_CHANGE = 50

    # Get expected timed state sequence according to the model (plan, problem, domain)
    (pddl_plan, pddl_problem, pddl_domain) = get_plan_problem_domain
    timed_state_seq = get_timed_state_seq(pddl_domain, pddl_problem, pddl_plan, DELTA_T)

    # Modify the model
    manipulator = ManipulateInitNumericFluent([GRAVITY],GRAVITY_CHANGE)
    manipulator.apply_change(pddl_domain, pddl_problem)

    # Get the new expected timed state sequence, according to the modified model
    modified_timed_state_seq = get_timed_state_seq(pddl_domain, pddl_problem, pddl_plan, DELTA_T)
    timed_state_seq = timed_state_seq[:len(modified_timed_state_seq)] # Trim to compare

    # Compute the diff between the timed state sequence and the not-timed state sequence. There should be none.
    diff_list = diff_traces(timed_state_seq, modified_timed_state_seq)
    logger.info("\n".join(diff_list))
    assert len(diff_list)>0 # Assert expected timed state sequence has changed due to the change in the model

    # Create a un-timed state sequence, to simulate how the SB observations would look like.
    modified_state_seq = [state for (state, t) in modified_timed_state_seq]

    # Assert the un-timed sequence created by the modified model is inconsistent with the timed sequence created by the original model
    consistency_checker = SingleNumericFluentConsistencyChecker(tuple([GRAVITY]))
    consistent_score = consistency_checker.estimate_consistency(timed_state_seq, modified_state_seq)
    assert consistent_score > 0.0


''' Test the single numeric fluent consistency checker, for the case where the numeric fluent is not constant,
 and the outcome of the tests should be that the sequences are consistent. '''
def test_single_constant_numeric_fluent_consistent(get_plan_problem_domain):
    DELTA_T = 0.05
    X_REDBIRD = ('y_bird', 'redbird_0')

    # Get expected timed and un-timed state sequence according to the model (plan, problem, domain)
    (pddl_plan, pddl_problem, pddl_domain) = get_plan_problem_domain
    timed_state_seq = get_timed_state_seq(pddl_domain, pddl_problem, pddl_plan, DELTA_T)
    state_seq = [state for (state, t) in timed_state_seq]

    # Check that the un-timed sequence is consistent with the timed sequence, w.r.t. the X_REDBIRD fluent
    consistency_checker = SingleNumericFluentConsistencyChecker(X_REDBIRD)
    assert consistency_checker.estimate_consistency(timed_state_seq, state_seq)<PRECISION

    # Check that subsets of the un-timed sequence are consistent with the timed sequence, w.r.t. the X_REDBIRD fluent
    partial_state_seq = state_seq[:(len(state_seq)-5)]
    assert consistency_checker.estimate_consistency(timed_state_seq, partial_state_seq)<PRECISION

    partial_state_seq = [state_seq[i] for i in range(0,len(state_seq),2)] # Get every other state
    assert consistency_checker.estimate_consistency(timed_state_seq, partial_state_seq)<PRECISION

    partial_state_seq = [state_seq[i] for i in range(0,len(state_seq),4)] # Get every forth state
    assert consistency_checker.estimate_consistency(timed_state_seq, partial_state_seq)<PRECISION


''' Test the single numeric fluent consistency checker, for the case where the numeric fluent is not constant,
 and the outcome of the tests should be that the sequences are inconsistent. '''
def test_single_numeric_fluent_inconsistent(get_plan_problem_domain):
    X_REDBIRD = ('y_bird', 'redbird_0')
    DELTA_T = 0.05
    GRAVITY_CHANGE = 50
    GRAVITY = "gravity"

    # Get expected timed state sequence according to the model (plan, problem, domain)
    (pddl_plan, pddl_problem, pddl_domain) = get_plan_problem_domain
    timed_state_seq = get_timed_state_seq(pddl_domain, pddl_problem, pddl_plan, DELTA_T)

    # Modify the model
    manipulator = ManipulateInitNumericFluent([GRAVITY],GRAVITY_CHANGE)
    manipulator.apply_change(pddl_domain, pddl_problem)

    # Get the new expected timed state sequence, according to the modified model
    modified_timed_state_seq = get_timed_state_seq(pddl_domain, pddl_problem, pddl_plan, DELTA_T)
    # Assert new timed state sequence is different from the original timed state sequence
    diff_list = diff_traces(timed_state_seq, modified_timed_state_seq)
    logger.info("\n".join(diff_list))
    assert len(diff_list)>0

    # Create a un-timed state sequence, to simulate how the SB observations would look like.
    modified_state_seq = [state for (state, t) in modified_timed_state_seq]

    # Assert the un-timed sequence created by the modified model is inconsistent with the timed sequence created by the original model
    consistency_checker = SingleNumericFluentConsistencyChecker(X_REDBIRD)
    consistent_score = consistency_checker.estimate_consistency(timed_state_seq, modified_state_seq)
    assert consistent_score > PRECISION


''' Returns a sequence of PDDL+ states outputted from SB '''
@pytest.fixture(scope="module")
def get_observed_state_seq():
    observations = []
    for i in range(0, 17):
        observations.append(sb.SBState.load_from_serialized_state(
            path.join(settings.ROOT_PATH, 'data', 'science_birds', 'serialized_levels', 'level-01', 'dx_test_{}.p'.format(i))))
    assert len(observations) == 17
    perception = Perception()
    state_seq = []
    bird_count = 100 # I'm assuming this is the maximum number of birds we will get.
    for observation in observations:
        if isinstance(observation.objects, list):
            perception.process_sb_state(observation)

        new_state = observation.translate_intermediate_state_to_pddl_state()
        state_seq.append(new_state)

        # This is a hack to identify when the observed state is from a new level
        birds  = new_state.get_birds()
        if len(birds)<bird_count:
            bird_count = len(birds)
        assert not(len(birds)>bird_count) # A new level has started TODO: Find a better way to identify that a new level has started
    return state_seq

''' Tests consistency check with real observations '''
def test_real_observations(get_plan_problem_domain, get_observed_state_seq):
    DELTA_T = 0.05
    Y_REDBIRD = ('y_bird', 'redbird_0')
    GRAVITY_CHANGE = -100
    GRAVITY = "gravity"

    observed_state_seq = get_observed_state_seq # Get the states observed from SB 

    # Get the states we expect to encounter while executing the plan according to our model
    (pddl_plan, pddl_problem, pddl_domain) = get_plan_problem_domain
    timed_state_seq = get_timed_state_seq(pddl_domain, pddl_problem, pddl_plan, DELTA_T)
    expected_state_seq = [timed_state[0] for timed_state in timed_state_seq]


    consistency_checker = SingleNumericFluentConsistencyChecker(Y_REDBIRD)
    consistent_score = consistency_checker.estimate_consistency(timed_state_seq, observed_state_seq)
    logger.info("Consistency score=%.2f" % consistent_score)
    assert consistent_score < PRECISION # The model is correct

    # Modify the model so it is incorrect
    manipulator = ManipulateInitNumericFluent([GRAVITY],GRAVITY_CHANGE)
    manipulator.apply_change(pddl_domain, pddl_problem)

    # Get the new expected timed state sequence, according to the modified model
    modified_timed_state_seq = get_timed_state_seq(pddl_domain, pddl_problem, pddl_plan, DELTA_T)
    # Assert new timed state sequence is different from the original timed state sequence
    diff_list = diff_traces(timed_state_seq, modified_timed_state_seq)
    logger.info("\n".join(diff_list))
    assert len(diff_list)>0

    # Assert the un-timed sequence created by the modified model is inconsistent with the timed sequence created by the original model
    diff = diff_pddl_states(timed_state_seq[0][0], observed_state_seq[0])

    consistency_checker = SingleNumericFluentConsistencyChecker(Y_REDBIRD)
    consistent_score = consistency_checker.estimate_consistency(timed_state_seq, observed_state_seq)
    logger.info("Consistency score=%.2f" % consistent_score)
    assert consistent_score > PRECISION