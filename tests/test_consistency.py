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

Y_BIRD_FLUENT = ('y_bird', 'redbird_0')
X_BIRD_FLUENT = ('x_bird', 'redbird_0')
DELTA_T = 0.05
GRAVITY_CHANGE = -50
GRAVITY = "gravity"

''' Helper function: simulate the given plan, on the given problem and domain.  '''
def __get_timed_state_seq(domain: PddlPlusDomain, problem:PddlPlusProblem, plan: PddlPlusPlan, delta_t:float):
    simulator = PddlPlusSimulator()
    (_, _, trace) =  simulator.simulate(plan, problem, domain, delta_t)
    return trace

''' Helper function: loads plan, problem, and domain used to evaluate consistency checker'''
def __load_plan_problem_domain():
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
def test_single_constant_numeric_fluent_consistent():
    # Get expected timed state sequence according to the model (plan, problem, domain)
    (pddl_plan, pddl_problem, pddl_domain) = __load_plan_problem_domain()
    timed_state_seq = __get_timed_state_seq(pddl_domain, pddl_problem, pddl_plan, DELTA_T)
    assert len(timed_state_seq)>0 # Sanity check

    # Create a un-timed state sequence, to simulate how the SB observations would look like.
    state_seq = [state for (state, t) in timed_state_seq]

    # Check that the un-timed state sequence is consistent with itself
    consistency_checker = SingleNumericFluentConsistencyChecker((GRAVITY))
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
    (pddl_plan, pddl_problem, pddl_domain) = __load_plan_problem_domain()
    timed_state_seq = __get_timed_state_seq(pddl_domain, pddl_problem, pddl_plan, DELTA_T)

    # Modify the model
    manipulator = ManipulateInitNumericFluent([GRAVITY],GRAVITY_CHANGE)
    manipulator.apply_change(pddl_domain, pddl_problem)

    # Get the new expected timed state sequence, according to the modified model
    modified_timed_state_seq = __get_timed_state_seq(pddl_domain, pddl_problem, pddl_plan, DELTA_T)
    timed_state_seq = timed_state_seq[:len(modified_timed_state_seq)] # Trim to compare

    # Compute the diff between the timed state sequence and the not-timed state sequence. There should be none.
    diff_list = diff_traces(timed_state_seq, modified_timed_state_seq)
    logger.info("\n".join(diff_list))
    assert len(diff_list)>0 # Assert expected timed state sequence has changed due to the change in the model

    # Create a un-timed state sequence, to simulate how the SB observations would look like.
    modified_state_seq = [state for (state, t) in modified_timed_state_seq]

    # Assert the un-timed sequence created by the modified model is inconsistent with the timed sequence created by the original model
    consistency_checker = SingleNumericFluentConsistencyChecker(X_BIRD_FLUENT)
    consistent_score = consistency_checker.estimate_consistency(timed_state_seq, modified_state_seq)
    assert consistent_score > 0.0


''' Test the single numeric fluent consistency checker, for the case where the numeric fluent is not constant,
 and the outcome of the tests should be that the sequences are consistent. '''
def test_single_constant_numeric_fluent_consistent():
    # Get expected timed and un-timed state sequence according to the model (plan, problem, domain)
    (pddl_plan, pddl_problem, pddl_domain) = __load_plan_problem_domain()
    timed_state_seq = __get_timed_state_seq(pddl_domain, pddl_problem, pddl_plan, DELTA_T)
    state_seq = [state for (state, t) in timed_state_seq]

    # Check that the un-timed sequence is consistent with the timed sequence, w.r.t. the X_REDBIRD fluent
    consistency_checker = SingleNumericFluentConsistencyChecker(Y_BIRD_FLUENT)
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
def test_single_numeric_fluent_inconsistent():
    # Get expected timed state sequence according to the model (plan, problem, domain)
    (pddl_plan, pddl_problem, pddl_domain) = __load_plan_problem_domain()
    timed_state_seq = __get_timed_state_seq(pddl_domain, pddl_problem, pddl_plan, DELTA_T)

    # Modify the model
    manipulator = ManipulateInitNumericFluent([GRAVITY],GRAVITY_CHANGE)
    manipulator.apply_change(pddl_domain, pddl_problem)

    # Get the new expected timed state sequence, according to the modified model
    modified_timed_state_seq = __get_timed_state_seq(pddl_domain, pddl_problem, pddl_plan, DELTA_T)
    # Assert new timed state sequence is different from the original timed state sequence
    diff_list = diff_traces(timed_state_seq, modified_timed_state_seq)
    logger.info("\n".join(diff_list))
    assert len(diff_list)>0

    # Create a un-timed state sequence, to simulate how the SB observations would look like.
    modified_state_seq = [state for (state, t) in modified_timed_state_seq]

    # Assert the un-timed sequence created by the modified model is inconsistent with the timed sequence created by the original model
    consistency_checker = SingleNumericFluentConsistencyChecker(Y_BIRD_FLUENT)
    consistent_score = consistency_checker.estimate_consistency(timed_state_seq, modified_state_seq)
    assert consistent_score > PRECISION


''' Test the single numeric fluent consistency checker, for the case where the numeric fluent is not constant,
 and the outcome of the tests should be that the sequences are inconsistent. '''
def test_multiple_numeric_fluent_inconsistent():
    # Get expected timed state sequence according to the model (plan, problem, domain)
    (pddl_plan, pddl_problem, pddl_domain) = __load_plan_problem_domain()
    timed_state_seq = __get_timed_state_seq(pddl_domain, pddl_problem, pddl_plan, DELTA_T)

    # Modify the model
    manipulator = ManipulateInitNumericFluent([GRAVITY],GRAVITY_CHANGE)
    manipulator.apply_change(pddl_domain, pddl_problem)

    # Get the new expected timed state sequence, according to the modified model
    modified_timed_state_seq = __get_timed_state_seq(pddl_domain, pddl_problem, pddl_plan, DELTA_T)
    # Assert new timed state sequence is different from the original timed state sequence
    diff_list = diff_traces(timed_state_seq, modified_timed_state_seq)
    logger.info("\n".join(diff_list))
    assert len(diff_list)>0

    # Create a un-timed state sequence, to simulate how the SB observations would look like.
    modified_state_seq = [state for (state, t) in modified_timed_state_seq]

    # Assert the un-timed sequence created by the modified model is inconsistent with the timed sequence created by the original model
    consistency_checker = NumericFluentsConsistencyChecker([X_BIRD_FLUENT,Y_BIRD_FLUENT])
    consistent_score = consistency_checker.estimate_consistency(timed_state_seq, modified_state_seq)
    assert consistent_score > PRECISION


''' Test the single numeric fluent consistency checker, for the case where the numeric fluent is not constant,
 and the outcome of the tests should be that the sequences are inconsistent. '''
def test_all_bird_numeric_fluent_inconsistent():
    # Get expected timed state sequence according to the model (plan, problem, domain)
    (pddl_plan, pddl_problem, pddl_domain) = __load_plan_problem_domain()
    timed_state_seq = __get_timed_state_seq(pddl_domain, pddl_problem, pddl_plan, DELTA_T)


    fluents_to_check = [fluent_name for fluent_name in PddlPlusState(pddl_problem.init).numeric_fluents.keys()
                        if "bird" in fluent_name[0].lower()]

    consistency_checker = NumericFluentsConsistencyChecker(fluents_to_check)
    # Assert the un-timed sequence created by the original model is consistent with the timed sequence created by the original model
    original_state_seq = [state for (state,t) in timed_state_seq]
    consistency_score = consistency_checker.estimate_consistency(timed_state_seq, original_state_seq)
    assert consistency_score < PRECISION


    # Modify the model
    manipulator = ManipulateInitNumericFluent([GRAVITY],GRAVITY_CHANGE)
    manipulator.apply_change(pddl_domain, pddl_problem)

    # Get the new expected timed state sequence, according to the modified model
    modified_timed_state_seq = __get_timed_state_seq(pddl_domain, pddl_problem, pddl_plan, DELTA_T)
    # Assert new timed state sequence is different from the original timed state sequence
    diff_list = diff_traces(timed_state_seq, modified_timed_state_seq)
    logger.info("\n".join(diff_list))
    assert len(diff_list)>0

    # Create a un-timed state sequence, to simulate how the SB observations would look like.
    modified_state_seq = [state for (state, t) in modified_timed_state_seq]
    # Assert the un-timed sequence created by the modified model is inconsistent with the timed sequence created by the original model
    consistent_score = consistency_checker.estimate_consistency(timed_state_seq, modified_state_seq)
    assert consistent_score > PRECISION


    consistent_score = consistency_checker.estimate_consistency(timed_state_seq, modified_state_seq)
    assert consistent_score > PRECISION

''' Loads a sequence of observed states of the given formal and desired quantity '''
def __load_observed_states(file_format, num_of_observations):
    observations = []
    for i in range(0, num_of_observations):
        observations.append(sb.SBState.load_from_serialized_state(file_format.format(i)))
    assert len(observations) == 17
    perception = Perception()
    state_seq = []
    bird_count = 100  # I'm assuming this is the maximum number of birds we will get.
    for observation in observations:
        if isinstance(observation.objects, list):
            perception.process_sb_state(observation)

        new_state = observation.translate_intermediate_state_to_pddl_state()
        state_seq.append(new_state)

        # This is a hack to identify when the observed state is from a new level
        birds = new_state.get_birds()
        if len(birds) < bird_count:
            bird_count = len(birds)
        assert not (len(
            birds) > bird_count)  # A new level has started TODO: Find a better way to identify that a new level has started
    return state_seq

def __print_fluents_values(state_seq:list, fluent_names:list):
    headers = ",".join([str(fluent_name) for fluent_name in fluent_names])
    print("\n%s" % headers)
    for state in state_seq:
        line = ",".join(["%.2f" % state[fluent_name] for fluent_name in fluent_names])
        print("%s" % line)

''' Tests consistency check with real observations '''
def test_real_observations():
    # Get the states observed from SB
    file_format = path.join(settings.ROOT_PATH, 'data', 'science_birds', 'serialized_levels', 'level-01', 'dx_test_{}.p')
    num_of_observations = 17
    observed_state_seq = __load_observed_states(file_format, num_of_observations)

    # Get the states we expect to encounter while executing the plan according to our model
    (pddl_plan, pddl_problem, pddl_domain) = __load_plan_problem_domain()
    timed_state_seq = __get_timed_state_seq(pddl_domain, pddl_problem, pddl_plan, DELTA_T)
    expected_state_seq = [timed_state[0] for timed_state in timed_state_seq]
    #
    # __print_fluents_values(observed_state_seq, [Y_BIRD_FLUENT,X_BIRD_FLUENT])
    # print("\n-----------------------------\n")
    # __print_fluents_values(expected_state_seq, [Y_BIRD_FLUENT,X_BIRD_FLUENT])

    consistency_checker = NumericFluentsConsistencyChecker([Y_BIRD_FLUENT, X_BIRD_FLUENT])
    inconsistency_score_good_model = consistency_checker.estimate_consistency(timed_state_seq, observed_state_seq[1:])
    logger.info("Consistency score good=%.2f" % inconsistency_score_good_model)

    # Modify the model so it is incorrect
    manipulator = ManipulateInitNumericFluent([GRAVITY],GRAVITY_CHANGE)
    manipulator.apply_change(pddl_domain, pddl_problem)

    # Get the new expected timed state sequence, according to the modified model
    modified_timed_state_seq = __get_timed_state_seq(pddl_domain, pddl_problem, pddl_plan, DELTA_T)
    # Assert new timed state sequence is different from the original timed state sequence
    diff_list = diff_traces(timed_state_seq, modified_timed_state_seq)
    logger.info("\n".join(diff_list))
    assert len(diff_list)>0

    # Assert the un-timed sequence created by the modified model is inconsistent with the timed sequence created by the original model
    diff = diff_pddl_states(timed_state_seq[0][0], observed_state_seq[0])

    __print_fluents_values([state for (state,t) in modified_timed_state_seq], [X_BIRD_FLUENT,Y_BIRD_FLUENT])


    consistency_checker = SingleNumericFluentConsistencyChecker(Y_BIRD_FLUENT)
    inconsistency_score_bad_model = consistency_checker.estimate_consistency(modified_timed_state_seq, observed_state_seq[1:])
    logger.info("Consistency score bad=%.2f" % inconsistency_score_bad_model)
    assert inconsistency_score_bad_model > inconsistency_score_good_model


