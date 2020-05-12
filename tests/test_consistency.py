import pytest
import worlds.science_birds as sb
from agent.perception.perception import Perception
from agent.consistency.consistency_estimator import *
from agent.consistency.pddl_plus_simulator import *
from agent.planning.planner import *
from agent.planning.pddlplus_parser import *
from agent.planning.model_manipulator import ManipulateInitNumericFluent
import logging
import worlds.science_birds as sb
from agent.hydra_agent import *
import matplotlib.pyplot as plt
import pickle
import tests.test_utils as test_utils

fh = logging.FileHandler("hydra.log",mode='w')
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger = logging.getLogger("test_consistency")
logger.setLevel(logging.INFO)
logger.addHandler(fh)

PRECISION = 1
DATA_DIR = path.join(settings.ROOT_PATH, 'data')
TRACE_DIR = path.join(DATA_DIR, 'science_birds', 'serialized_levels', 'level-01')
GRAVITY_BAD_OBS = path.join(TRACE_DIR,"g_250_observed.p")

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


''' Loads a sequence of observed states of the given formal and desired quantity '''
def _load_observed_states(file_format, num_of_observations, meta_model = MetaModel()):
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
        new_state = meta_model.create_pddl_state(observation)
        state_seq.append(new_state)

        # This is a hack to identify when the observed state is from a new level
        birds = new_state.get_birds()
        if len(birds) < bird_count:
            bird_count = len(birds)
        assert not (len(
            birds) > bird_count)  # A new level has started TODO: Find a better way to identify that a new level has started
    return state_seq

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
    consistency_checker = NumericFluentsConsistencyEstimator([X_BIRD_FLUENT, Y_BIRD_FLUENT])
    consistent_score = consistency_checker.estimate_consistency(timed_state_seq, modified_state_seq)
    assert consistent_score > PRECISION


''' Test the single numeric fluent consistency checker, for the case where the numeric fluent is not constant,
 and the outcome of the tests should be that the sequences are inconsistent. '''
def test_all_bird_numeric_fluent_inconsistent():
    # Get expected timed state sequence according to the model (plan, problem, domain)
    (plan, problem, domain) = _load_plan_problem_domain()
    simulation_trace = test_utils.simulate_plan_trace(plan, problem, domain, DELTA_T)


    fluents_to_check = [fluent_name for fluent_name in PddlPlusState(problem.init).numeric_fluents.keys()
                        if "bird" in fluent_name[0].lower()]

    consistency_checker = NumericFluentsConsistencyEstimator(fluents_to_check)
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


# TODO: This data is obselete, considering removing this test
''' Tests consistency check with real observations '''
def test_real_observations():
    # Get the states observed from SB
    file_format = path.join(settings.ROOT_PATH, 'data', 'science_birds', 'serialized_levels', 'level-01', 'dx_test_{}.p')
    num_of_observations = 17
    observed_state_seq = _load_observed_states(file_format, num_of_observations)

    # Get the states we expect to encounter while executing the plan according to our model
    (plan, problem, domain) = _load_plan_problem_domain()
    simulated_plan_trace = test_utils.simulate_plan_trace(plan, problem, domain, DELTA_T)

    consistency_checker = NumericFluentsConsistencyEstimator([Y_BIRD_FLUENT, X_BIRD_FLUENT])
    test_utils.plot_bird_xy_series([state for (state,_,_) in simulated_plan_trace[1:3]], observed_state_seq)
    inconsistency_score_good_model = consistency_checker.estimate_consistency(simulated_plan_trace, observed_state_seq[1:3])
    logger.info("Consistency score good=%.2f" % inconsistency_score_good_model)

    # Modify the model so it is incorrect
    manipulator = ManipulateInitNumericFluent([GRAVITY],GRAVITY_CHANGE)
    manipulator.apply_change(domain, problem)

    # Get the new expected timed state sequence, according to the modified model
    modified_timed_state_seq = test_utils.simulate_plan_trace(plan, problem, domain, DELTA_T)

    # Assert new timed state sequence is different from the original timed state sequence
    diff_list = diff_traces(simulated_plan_trace, modified_timed_state_seq)
    logger.info("\n".join(diff_list))
    assert len(diff_list)>0

    # Assert the un-timed sequence created by the modified model is inconsistent with the timed sequence created by the original model
    diff = diff_pddl_states(simulated_plan_trace[0][0], observed_state_seq[0])

    # __print_fluents_values([state for (state,t) in modified_timed_state_seq], [X_BIRD_FLUENT,Y_BIRD_FLUENT])


    consistency_checker = SingleNumericFluentConsistencyEstimator(Y_BIRD_FLUENT)
    inconsistency_score_bad_model = consistency_checker.estimate_consistency(modified_timed_state_seq, observed_state_seq[1:])
    logger.info("Consistency score bad=%.2f" % inconsistency_score_bad_model)
    assert inconsistency_score_bad_model > inconsistency_score_good_model


#################### System Test ###############################
@pytest.fixture(scope="module")
def launch_science_birds():
    logger.info("starting")
    #remove config files
    cmd = 'cp {}/data/science_birds/level-14.xml {}/00001.xml'.format(str(settings.ROOT_PATH), str(settings.SCIENCE_BIRDS_LEVELS_DIR))
    subprocess.run(cmd, shell=True)
    cmd = 'cp {}/data/science_birds/level-15.xml {}/00002.xml'.format(str(settings.ROOT_PATH), str(settings.SCIENCE_BIRDS_LEVELS_DIR))
    subprocess.run(cmd, shell=True)
    cmd = 'cp {}/data/science_birds/level-16.xml {}/00003.xml'.format(str(settings.ROOT_PATH), str(settings.SCIENCE_BIRDS_LEVELS_DIR))
    subprocess.run(cmd, shell=True)
    logger.info("Launching ScienceBirds...")
    env = sb.ScienceBirds(None,launch=True)
    logger.info("ScienceBirds launched!")
    yield env
    logger.info("teardown tests")
    env.kill()



''' Run Hydra, collect observations, check for consistency '''
@pytest.mark.skipif(settings.HEADLESS == True, reason="headless does not work in docker")
def test_consistency_in_agent(launch_science_birds):
    env = launch_science_birds
    hydra = HydraAgent(env)

    # Inject fault and play
    original_gravity = hydra.meta_model.constant_numeric_fluents[GRAVITY]
    hydra.meta_model.constant_numeric_fluents[GRAVITY] = 250.0
    hydra.main_loop(max_actions=3)  # enough actions to play the first level

    game_state = env.sb_client.get_game_state()
    assert game_state.value != GameState.WON.value

    # Get the state and the action, simulate expected observations
    # Extract intermediate states, compate with simulated expected observations

    our_observation =  hydra.observations[1]
    pickle.dump(our_observation,open(GRAVITY_BAD_OBS, "wb"))

    (expected_timed_seq, observed_seq) = _extract_expected_and_observed(hydra.meta_model, our_observation)
    test_utils.plot_bird_xy_series([state[0] for state in expected_timed_seq], observed_seq)

    hydra.meta_model.constant_numeric_fluents[GRAVITY] = original_gravity
    (expected_timed_seq, observed_seq) = _extract_expected_and_observed(hydra.meta_model, our_observation)
    test_utils.plot_bird_xy_series([state[0] for state in expected_timed_seq], observed_seq)
    print("All done")

''' Run Hydra agent with a planner that has a wrong meta model, which assumes (erronously) that the gravity is 250. '''
def _create_gravity_250_observation():
    env = launch_science_birds
    hydra = HydraAgent(env)

    # Inject fault and play
    hydra.meta_model.constant_numeric_fluents[GRAVITY] = 250.0
    hydra.main_loop(max_actions=3)  # enough actions to play the first level

    our_observation =  hydra.observations[1]
    pickle.dump(our_observation,open(GRAVITY_BAD_OBS, "wb"))

def test_gravity_250():
    meta_model = MetaModel()

    # Inject fault and play
    original_gravity = meta_model.constant_numeric_fluents[GRAVITY]
    meta_model.constant_numeric_fluents[GRAVITY] = 250.0

    # Load observation as created by the _create_gravity_250_observation() function
    our_observation = pickle.load(open(GRAVITY_BAD_OBS, "rb"))
    (expected_timed_seq, observed_seq) = _extract_expected_and_observed(meta_model, our_observation)
    test_utils.plot_bird_xy_series([state[0] for state in expected_timed_seq],
                                   observed_seq)

    meta_model.constant_numeric_fluents[GRAVITY] = original_gravity
    (expected_timed_seq, observed_seq) = _extract_expected_and_observed(meta_model, our_observation)
    test_utils.plot_bird_xy_series([state[0] for state in expected_timed_seq],
                         observed_seq)

    print(3)


''' Extract from the given observation Hydra collected, an expected list of itnermediate states and an observed y'''
def _extract_expected_and_observed(meta_model: MetaModel, our_observation: ScienceBirdsObservation):
    assert our_observation.action is not None
    assert our_observation.state is not None

    problem = meta_model.create_pddl_problem(our_observation.state)
    domain = meta_model.create_pddl_domain(our_observation.state)
    planner = Planner()
    grounded_domain = PddlPlusGrounder().ground_domain(domain, problem)  # Needed to identify plan action
    plan_trace_file = "%s/docker_plan_trace.txt" % str(settings.PLANNING_DOCKER_PATH)
    plan = planner.extract_plan_from_plan_trace(plan_trace_file, grounded_domain)


    expected_timed_state_seq = test_utils.simulate_plan_trace(plan, problem, domain, DELTA_T)
    observed_state_seq = []
    perception = Perception()
    for intermediate_state in our_observation.intermediate_states:
        if isinstance(intermediate_state.objects, list):
            intermediate_state = perception.process_sb_state(intermediate_state)
        observed_state_seq.append(meta_model.create_pddl_state(intermediate_state))

    return (expected_timed_state_seq, observed_state_seq)