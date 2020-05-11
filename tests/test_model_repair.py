import pytest
import os.path as path
import settings
from agent.planning.pddlplus_parser import PddlProblemParser, PddlProblemExporter, PddlDomainParser, PddlDomainExporter
from agent.planning.model_manipulator import ManipulateInitNumericFluent
from agent.planning.planner import Planner
from agent.consistency.model_repair import *
from agent.consistency.meta_model_repair import *
from agent.planning.pddl_meta_model import *
import agent.planning.pddl_plus as pddl_plus
from worlds.science_birds import SBState
from agent.perception.perception import Perception
from worlds.science_birds import ScienceBirds as sb
import subprocess
import worlds.science_birds as sb
from agent.hydra_agent import *


DATA_DIR = path.join(settings.ROOT_PATH, 'data')
TRACE_DIR = path.join(DATA_DIR, 'science_birds', 'serialized_levels', 'level-01')
PRECISION = 0.0001
DELTA_T = 0.05
GRAVITY = ["gravity"]
GRAVITY_STR = "gravity"

X_REDBIRD = ('y_bird', 'redbird_0')

''' Helper function: get the gravity value from the initial state '''
def __get_gravity_value_in_init(pddl_problem):
    gravity_fluent = pddl_plus.get_numeric_fluent(pddl_problem.init, GRAVITY)
    return float(pddl_plus.get_numeric_fluent_value(gravity_fluent))

''' Helper function: simulate the observed trace, for testing. TODO: Replace this with real traces '''
def __simulate_observed_trace(domain: PddlPlusDomain, problem:PddlPlusProblem, plan: PddlPlusPlan, delta_t:float):
    simulator = PddlPlusSimulator()
    (_,_,timed_state_seq) =  simulator.simulate(plan, problem, domain, delta_t)
    return timed_state_seq

''' Helper function: loads plan, problem, and domain from files to evaluate consistency checker'''
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

''' Test changing a single numeric fluent, and the ability of model repair to repair it'''
def test_single_numeric_repair():

    # Get expected timed state sequence according to the model (plan, problem, domain)
    (pddl_plan, pddl_problem, pddl_domain) = __load_plan_problem_domain()
    observed_trace = __simulate_observed_trace(pddl_domain, pddl_problem, pddl_plan, DELTA_T)
    observed_state_seq = [timed_state[0] for timed_state in observed_trace]

    # Apply the change
    original_gravity_value = __get_gravity_value_in_init(pddl_problem)
    gravity_delta = 1
    manipulator = ManipulateInitNumericFluent(GRAVITY,3*gravity_delta)
    manipulator.apply_change(pddl_domain, pddl_problem)

    new_gravity_value = __get_gravity_value_in_init(pddl_problem)
    assert new_gravity_value, original_gravity_value+3*gravity_delta # Sanity check: manipulator working

    expected_timed_trace = __simulate_observed_trace(pddl_domain, pddl_problem, pddl_plan, DELTA_T)
    model_repair = SingleNumericFluentRepair(GRAVITY, X_REDBIRD, gravity_delta)
    assert model_repair.is_consistent(expected_timed_trace, observed_state_seq) == False

    # Apply the model repair algorithm
    model_repair = SingleNumericFluentRepair(GRAVITY, X_REDBIRD, gravity_delta)
    model_repair.repair(pddl_domain, pddl_problem, pddl_plan, observed_state_seq)

    # Assert repair was able to restore the gravity to its correct value
    repaired_gravity_value = __get_gravity_value_in_init(pddl_problem)
    assert abs(original_gravity_value - repaired_gravity_value)<PRECISION


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



''' Test scenario generated by running a plan with gravity 250 '''
def test_repair_gravity_250():
    meta_model = MetaModel()
    original_gravity = meta_model.constant_numeric_fluents['gravity']
    bad_gravity = 250.0
    meta_model.constant_numeric_fluents['gravity'] = bad_gravity # Fault injection

    sb_state_file = path.join(TRACE_DIR, "state_0.p")
    sb_state = SBState.load_from_serialized_state(sb_state_file)

    problem_file = path.join(TRACE_DIR, "sb_prob.pddl")
    problem_parser = PddlProblemParser()
    pddl_problem = problem_parser.parse_pddl_problem(problem_file)
    assert pddl_problem is not None, "PDDL+ problem object not parsed"  # Sanity check, parser works

    domain_file = path.join(TRACE_DIR, "sb_domain.pddl")
    domain_parser = PddlDomainParser()
    pddl_domain = domain_parser.parse_pddl_domain(domain_file)
    assert pddl_domain is not None, "PDDL+ domain object not parsed"  # Sanity check, parser works

    planner = Planner()
    grounded_domain = PddlPlusGrounder().ground_domain(pddl_domain, pddl_problem)  # Needed to identify plan action
    plan_trace_file = path.join(TRACE_DIR, "g_250_docker_plan_trace.txt")
    pddl_plan = planner.extract_plan_from_plan_trace(plan_trace_file, grounded_domain)
    expected_timed_state_seq = __simulate_observed_trace(pddl_domain, pddl_problem, pddl_plan, delta_t=0.05)
    assert len(expected_timed_state_seq)>0

    # Get observations
    perception = Perception()
    obs_state_seq = []
    for i in range(0, 12):
        obs = SBState.load_from_serialized_state(path.join(TRACE_DIR, 'repair-{}.p'.format(i)))
        if isinstance(obs.objects, list):
            obs = perception.process_sb_state(obs)
        obs_state = meta_model.create_pddl_state(obs)
        obs_state_seq.append(obs_state)

    assert len(obs_state_seq) == 12

    gravity_delta = 1
    meta_model_repair = MetaModelSingleNumericFluentRepair(GRAVITY_STR, X_REDBIRD, gravity_delta)
    assert meta_model_repair.is_consistent(expected_timed_state_seq, obs_state_seq) == False

    # Repair
    meta_model_repair.repair(meta_model, sb_state, pddl_plan, obs_state_seq, DELTA_T)

    # Assert repair was able to restore the gravity to its correct value
    repaired_gravity_value = meta_model.constant_numeric_fluents[GRAVITY_STR]
    assert abs(original_gravity - repaired_gravity_value) < PRECISION

#################### System tests ########################

@pytest.fixture(scope="module")
def launch_science_birds(config_file):
    logger.info("starting")
    #remove config files
    cmd = 'cp {}/data/science_birds/level-14.xml {}/00001.xml'.format(str(settings.ROOT_PATH), str(settings.SCIENCE_BIRDS_LEVELS_DIR))
    subprocess.run(cmd, shell=True)
    cmd = 'cp {}/data/science_birds/level-15.xml {}/00002.xml'.format(str(settings.ROOT_PATH), str(settings.SCIENCE_BIRDS_LEVELS_DIR))
    subprocess.run(cmd, shell=True)
    cmd = 'cp {}/data/science_birds/level-16.xml {}/00003.xml'.format(str(settings.ROOT_PATH), str(settings.SCIENCE_BIRDS_LEVELS_DIR))
    subprocess.run(cmd, shell=True)
    logger.info("Launching ScienceBirds...")
    env = sb.ScienceBirds(None,launch=True,config_file=config_file)
    logger.info("ScienceBirds launched!")
    yield env
    logger.info("teardown tests")
    env.kill()


''' A sanity check: running the first level with a bad gravity, changing it manually to a good gravity,
and verifying it is working after the change. '''
#@pytest.mark.skipif(settings.HEADLESS == True, reason="headless does not work in docker")
@pytest.mark.skipif(True, reason="headless does not work in docker")
def test_manual_repair_in_agent(launch_science_birds):
    config_file = '{}/data/science_birds/config/test_config_repeating.xml'.format(settings.ROOT_PATH)
    env = launch_science_birds(config_file)
    hydra = HydraAgent(env)

    # Inject fault and play
    original_gravity = hydra.meta_model.constant_numeric_fluents['gravity']
    hydra.meta_model.constant_numeric_fluents['gravity'] = 250.0
    hydra.main_loop(max_actions=3)  # enough actions to play the first level

    game_state = env.sb_client.get_game_state()
    assert game_state.value != GameState.WON.value

    hydra.meta_model.constant_numeric_fluents['gravity'] = original_gravity
    hydra.main_loop(max_actions=2)  # enough actions to play the first level

    game_state = env.sb_client.get_game_state()
    assert game_state.value == GameState.WON.value


''' A full system test: run SB with a bad meta model, observe results, fix meta model '''
#@pytest.mark.skipif(settings.HEADLESS == True, reason="headless does not work in docker")
@pytest.mark.skipif(True, reason="headless does not work in docker")
def test_repair_in_agent(launch_science_birds):
    env = launch_science_birds
    hydra = HydraAgent(env)

    # Inject fault and play
    original_gravity = hydra.meta_model.constant_numeric_fluents['gravity']
    hydra.meta_model.constant_numeric_fluents['gravity'] = 250.0
    hydra.main_loop(max_actions=6)  # enough actions to play the first level
    scores = env.get_all_scores()
    assert sum(scores) == 0  # Should fail if gravity is wrong

    # Get observed states
    obs_state_seq = []
    for state in env.intermediate_states:
        state = hydra.perception.process_state(state)
        pddl_state = hydra.meta_model.create_pddl_state(state)
        obs_state_seq.append(pddl_state)

    # Get plan
    plan_trace_file = "%s/docker_plan_trace.txt" % str(settings.PLANNING_DOCKER_PATH) # This is where the planner writes its plan
    planner = Planner()
    last_state = hydra.state_history[-1]
    domain = hydra.meta_model.create_pddl_domain(last_state)
    problem = hydra.meta_model.create_pddl_problem(last_state)
    plan = planner.extract_plan_from_plan_trace(plan_trace_file, domain)
    expected_timed_state_seq = __simulate_observed_trace(domain, problem, plan, delta_t=0.05)

    gravity_delta = 1
    meta_model_repair = MetaModelSingleNumericFluentRepair(GRAVITY_STR, X_REDBIRD, gravity_delta)
    assert meta_model_repair.is_consistent(expected_timed_state_seq, obs_state_seq) == False

    meta_model_repair.repair(hydra.meta_model, plan, obs_state_seq,delta_t=0.05)

    correct_gravity =134.2
    assert abs(hydra.meta_model.constant_numeric_fluents['gravity']-correct_gravity)<5

    hydra.main_loop(max_actions=2)  # enough actions to play the first level
    scores = env.get_all_scores()
    assert sum(scores) > 0  # Should fail if gravity is wrong
