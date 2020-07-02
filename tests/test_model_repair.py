import logging
import subprocess

import matplotlib
import pytest

import agent.planning.pddl_plus as pddl_plus
import tests.test_utils as test_utils
from agent.consistency.meta_model_repair import *
from agent.hydra_agent import *
from agent.planning.planner import *
from agent.planning.simple_planner import *
import worlds.science_birds as sb

fh = logging.FileHandler("test_model_repair.log",mode='a')
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger = logging.getLogger("test_model_repair")
logger.setLevel(logging.DEBUG)
logger.addHandler(fh)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(fh)


DATA_DIR = path.join(settings.ROOT_PATH, 'data')
TRACE_DIR = path.join(DATA_DIR, 'science_birds', 'serialized_levels', 'level-01')
PRECISION = 0.0001
DELTA_T = 0.05
GRAVITY = ["gravity"]
GRAVITY_STR = "gravity"

X_REDBIRD = ('x_bird', 'redbird_0')
Y_REDBIRD = ('y_bird', 'redbird_0')


''' Helper function: get the gravity value from the initial state '''
def __get_gravity_value_in_init(pddl_problem):
    gravity_fluent = pddl_plus.get_numeric_fluent(pddl_problem.init, GRAVITY)
    return float(pddl_plus.get_numeric_fluent_value(gravity_fluent))

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
    pddl_plan = test_utils.extract_plan_from_plan_trace(plan_trace_file, grounded_domain)
    return (pddl_plan, pddl_problem, pddl_domain)


''' Helper function: Checks if the observation is consistency with executing the plan under the given meta model'''
def _check_consistency(observation: ScienceBirdsObservation,
                       plan: PddlPlusPlan,
                       meta_model: MetaModel,
                       consistency_checker = ConsistencyChecker,
                       delta_t : float = DELTA_T):
    problem = meta_model.create_pddl_problem(observation.state)
    domain = meta_model.create_pddl_domain(observation.state)
    expected_trace = test_utils.simulate_plan_trace(plan, problem, domain, delta_t)
    observed_seq = observation.get_trace(meta_model)
    return consistency_checker.estimate_consistency(expected_trace,observed_seq)


#################### System tests ########################
@pytest.fixture(scope="module")
def launch_science_birds():
    print("starting")
    env = sb.ScienceBirds(None,launch=True,config='test_consistency_config.xml')
    yield env
    print("teardown tests")
    env.kill()

''' A full system test: run SB with a bad meta model, observe results, fix meta model '''
@pytest.mark.skipif(settings.HEADLESS == True, reason="headless does not work in docker")
def test_repair_in_agent(launch_science_birds):
    DELTA_T = 1.0
    env = launch_science_birds
    hydra = HydraAgent(env)

    # Inject fault and play
    meta_model = hydra.meta_model
    meta_model.constant_numeric_fluents["ground_damper"] = 0.3
    meta_model.constant_numeric_fluents["angle_rate"] = 0.5
    fluents_to_repair = ("angle_rate",)
    repair_deltas = (0.1,)

    # meta_model.constant_numeric_fluents[ANGLE_RATE] = 2
    logger.info("Running agent with current meta model")
    hydra.planner = Planner(meta_model)
    hydra.main_loop(max_actions=3)  # enough actions to play the first level

    scores = env.get_all_scores()
    assert sum(scores) == 0  # Should fail if angle rate is wrong

    logger.info("Agent performed action %s " % str(hydra.observations[1].action))
    logger.info("Agent score: %d" % sum(scores))

    # Extract observed states
    observation = _find_last_obs(hydra)
    observed_seq = observation.get_trace(hydra.meta_model)

    obs_output_file = path.join(DATA_DIR, "obs_zero_temp3.p") # For debug
    pickle.dump(observation, open(obs_output_file, "wb"))  # For debug
    matplotlib.interactive(True) # For debug
    test_utils.plot_observation(observation) # For debug
    test_utils.plot_expected_trace(meta_model, observation.state, observation.action, delta_t=DELTA_T) # For debug

    # Compute expected trace of timed states
    problem = meta_model.create_pddl_problem(observation.state)
    domain = meta_model.create_pddl_domain(observation.state)
    grounded_domain = PddlPlusGrounder().ground_domain(domain,problem)
    plan = PddlPlusPlan()
    plan.add_raw_actions([observation.action], grounded_domain)
    expected_trace = test_utils.simulate_plan_trace(plan, problem, domain, delta_t=DELTA_T)

    # Check consistency
    consistency_fluents = [X_REDBIRD, Y_REDBIRD]
    consistency_checker = NumericFluentsConsistencyEstimator(consistency_fluents)
    consistency_before_repair = consistency_checker.estimate_consistency(expected_trace, observed_seq,delta_t=DELTA_T)
    logger.info("Consistency with model: %.2f " % consistency_before_repair)
    desired_precision = 30
    assert consistency_before_repair > desired_precision

    # Now apply repair
    logger.info("Starting model repair for fluents %s..." % str(fluents_to_repair))
    meta_model_repair = GreedyBestFirstSearchMetaModelRepair(fluents_to_repair,consistency_checker,repair_deltas)
    assert meta_model_repair.is_consistent(expected_trace, observed_seq) == False
    repaired_meta_model = meta_model_repair.repair(meta_model, observation.state, plan, observed_seq,delta_t=DELTA_T)
    logger.info("Repair done. Fluent values in meta model  are now %s" %
                str([repaired_meta_model.constant_numeric_fluents[fluent] for fluent in fluents_to_repair]))

    consistency_after_repair = _check_consistency(observation, plan, repaired_meta_model,consistency_checker, delta_t=DELTA_T)
    assert consistency_before_repair > consistency_after_repair
    assert consistency_after_repair < desired_precision
    logger.info("Consistency with model after repair: %.2f " % consistency_after_repair)

    logger.info("Running agent with repaired model...")
    hydra.planner.meta_model = repaired_meta_model
    hydra.main_loop(max_actions=1)  # enough actions to play the first level

    observation = _find_last_obs(hydra)
    plan = PddlPlusPlan()
    plan.add_raw_actions([observation.action], grounded_domain)

    logger.info("Agent performed action %s " % str(hydra.observations[1].action))
    logger.info("Agent score: %d" % sum(scores))

    scores = env.get_all_scores()
    assert sum(scores) > 0  # Should succeed if angle rate is wrong
    logger.info("Problem solved!")

@pytest.fixture(scope="module")
def launch_science_birds_level_01():
    print("starting")
    cmd = 'cp {}/data/science_birds/level-04.xml {}/00001.xml'.format(str(settings.ROOT_PATH), str(settings.SCIENCE_BIRDS_LEVELS_DIR))
    subprocess.run(cmd, shell=True)

    env = sb.ScienceBirds(None,launch=True,config='test_consistency_config.xml')
    yield env
    print("teardown tests")
    env.kill()

''' Repair gravity based on an observed state'''
def test_repair_gravity_offline():
    meta_model = MetaModel()
    fluents_to_repair = ["gravity_factor",]
    repair_deltas = [2.5,]
    desired_precision = 20
    delta_t = 0.05
    meta_model.constant_numeric_fluents["gravity_factor"] = 2.5
    obs_output_file = path.join(DATA_DIR, 'science_birds','tests', "bad_gravity_level_004.p") # For debug
    observation = pickle.load(open(obs_output_file, "rb"))  # For debug
    time_action = [observation.action[0], observation.action[1]/meta_model.get_angle_rate()]

    consistency_checker = BirdLocationConsistencyEstimator()
    meta_model_repair = FixedPointMetaModelRepair(fluents_to_repair, consistency_checker, repair_deltas,
                                                             consistency_threshold=desired_precision)

    problem = meta_model.create_pddl_problem(observation.state)
    domain = meta_model.create_pddl_domain(observation.state)
    grounded_domain = PddlPlusGrounder().ground_domain(domain,problem)
    plan = PddlPlusPlan()
    plan.add_raw_actions([time_action], grounded_domain)
    observed_seq = observation.get_trace(meta_model)

    consistency_before_repair = consistency_checker.estimate_consistency(test_utils.simulate_plan_trace(plan, problem, domain), observed_seq)

    repaired_meta_model = meta_model_repair.repair(meta_model, observation.state, plan, observed_seq, delta_t=delta_t)
    logger.info("Repair done. Fluent values in meta model  are now %s" %
                str([repaired_meta_model.constant_numeric_fluents[fluent] for fluent in fluents_to_repair]))

    problem = meta_model.create_pddl_problem(observation.state)
    domain = meta_model.create_pddl_domain(observation.state)
    consistency_after_repair = consistency_checker.estimate_consistency(test_utils.simulate_plan_trace(plan, problem, domain), observed_seq)

    assert consistency_before_repair>consistency_after_repair

''' Repair gravity based on an observed state'''
def test_repair_pig_bounce():
    meta_model = MetaModel()

    fluents_to_repair = ["pig_bounce_factor",]
    repair_deltas = [2.0,]
    desired_precision = 20
    delta_t = 0.05
    meta_model.constant_numeric_fluents[fluents_to_repair[0]] = 2.0
    obs_output_file = path.join(DATA_DIR, 'science_birds','tests', "bad_pig_bounce.p") # For debug
    observation = pickle.load(open(obs_output_file, "rb"))  # For debug
    time_action = [observation.action[0], observation.action[1]/meta_model.get_angle_rate()]

    consistency_checker = BirdLocationConsistencyEstimator()
    meta_model_repair = FixedPointMetaModelRepair(fluents_to_repair, consistency_checker, repair_deltas,
                                                             consistency_threshold=desired_precision)

    problem = meta_model.create_pddl_problem(observation.state)
    domain = meta_model.create_pddl_domain(observation.state)
    grounded_domain = PddlPlusGrounder().ground_domain(domain,problem)
    plan = PddlPlusPlan()
    plan.add_raw_actions([time_action], grounded_domain)
    observed_seq = observation.get_trace(meta_model)

    test_utils.plot_observation(observation) # For debug
    test_utils.plot_expected_trace(meta_model, observation.state, time_action, delta_t=DELTA_T) # For debug

    consistency_before_repair = consistency_checker.estimate_consistency(test_utils.simulate_plan_trace(plan, problem, domain), observed_seq)

    repaired_meta_model = meta_model_repair.repair(meta_model, observation.state, plan, observed_seq, delta_t=delta_t)
    logger.info("Repair done. Fluent values in meta model  are now %s" %
                str([repaired_meta_model.constant_numeric_fluents[fluent] for fluent in fluents_to_repair]))

    problem = meta_model.create_pddl_problem(observation.state)
    domain = meta_model.create_pddl_domain(observation.state)
    consistency_after_repair = consistency_checker.estimate_consistency(test_utils.simulate_plan_trace(plan, problem, domain), observed_seq)

    assert consistency_before_repair>consistency_after_repair


''' A full system test: run SB with a bad meta model, observe results, fix meta model '''
@pytest.mark.skipif(settings.HEADLESS == True, reason="headless does not work in docker")
def test_repair_gravity_in_agent(launch_science_birds_level_01):
    DELTA_T = 0.05
    desired_precision = 20

    env = launch_science_birds_level_01
    hydra = HydraAgent(env)

    # Inject fault and play
    meta_model = hydra.meta_model

    fluents_to_repair = ["gravity_factor",]
    repair_deltas = [2.5, ]
    meta_model.constant_numeric_fluents["gravity_factor"] = 2.5


    logger.info("Running agent with current meta model")

    # Run agent with dummy action and collect observation
    # raw_timed_action = ["pa-twang redbird_0", 65.5]
    # plan = [raw_timed_action]
    # hydra.planner = test_utils.PlannerStub(plan, hydra.meta_model)

    hydra.planner = Planner(meta_model)
    hydra.main_loop(max_actions=3)  # enough actions to play the first level
    logger.info("Agent performed action %s " % str(hydra.observations[1].action))

    observation = _find_last_obs(hydra)
    assert observation.reward == 0

    # Extract expected and observed states
    observed_seq = observation.get_trace(hydra.meta_model)
    obs_output_file = path.join(DATA_DIR, "bad_gravity_level_004.p") # For debug
    pickle.dump(observation, open(obs_output_file, "wb"))  # For debug
    # matplotlib.interactive(True) # For debug
    # test_utils.plot_observation(observation) # For debug
    time_action = [observation.action[0], observation.action[1]/meta_model.get_angle_rate()]
    # test_utils.plot_expected_trace(meta_model, observation.state, time_action, delta_t=DELTA_T) # For debug
    problem = meta_model.create_pddl_problem(observation.state)
    domain = meta_model.create_pddl_domain(observation.state)
    grounded_domain = PddlPlusGrounder().ground_domain(domain,problem)
    plan = PddlPlusPlan()
    plan.add_raw_actions([time_action], grounded_domain)
    expected_trace = test_utils.simulate_plan_trace(plan, problem, domain, delta_t=DELTA_T)

    # Check consistency
    consistency_checker = BirdLocationConsistencyEstimator()
    consistency_before_repair = consistency_checker.estimate_consistency(expected_trace, observed_seq, delta_t=DELTA_T)
    logger.info("Consistency with model: %.2f " % consistency_before_repair)
    assert consistency_before_repair > desired_precision

    # Now apply repair
    logger.info("Starting model repair for fluents %s..." % str(fluents_to_repair))
    meta_model_repair = FixedPointMetaModelRepair(fluents_to_repair, consistency_checker, repair_deltas, consistency_threshold=desired_precision)
    assert meta_model_repair.is_consistent(expected_trace, observed_seq) == False
    repaired_meta_model = meta_model_repair.repair(meta_model, observation.state, plan, observed_seq, delta_t=DELTA_T)
    logger.info("Repair done. Fluent values in meta model  are now %s" %
                str([repaired_meta_model.constant_numeric_fluents[fluent] for fluent in fluents_to_repair]))

    consistency_after_repair = _check_consistency(observation, plan, repaired_meta_model, consistency_checker,
                                                  delta_t=DELTA_T)
    assert consistency_before_repair > consistency_after_repair
    logger.info("Consistency with model after repair: %.2f " % consistency_after_repair)

    logger.info("Running agent with repaired model...")
    hydra.planner.meta_model = repaired_meta_model
    _run_next_action(hydra)  # enough actions to play a level

    observation = _find_last_obs(hydra)
    assert observation.reward > 0

    # Play next level

    # Re-inject fault
    meta_model.constant_numeric_fluents["gravity_factor"] = 2.5

    _run_next_action(hydra)  # enough actions to play a level
    meta_model = repaired_meta_model
    meta_model_repair = FixedPointMetaModelRepair(fluents_to_repair, consistency_checker, repair_deltas,
                                                             consistency_threshold=desired_precision)
    observation = _find_last_obs(hydra)

    obs_output_file = path.join(DATA_DIR, "bad_gravity_2.p") # For debug
    pickle.dump(observation, open(obs_output_file, "wb"))  # For debug

    observed_seq = observation.get_trace(hydra.meta_model)
    repaired_meta_model = meta_model_repair.repair(meta_model, observation.state, plan, observed_seq, delta_t=DELTA_T)
    hydra.planner.meta_model = repaired_meta_model

    # Play next level with repaired model
    _run_next_action(hydra)  # enough actions to play a level
    observation = _find_last_obs(hydra)
    assert observation.reward > 0


def _run_next_action(hydra_agent: HydraAgent):
    while True:
        hydra_agent.main_loop(max_actions=1)
        if hydra_agent.observations[-1].action is not None:
            return

''' Finds the last observations of the game. That is, the last observation that has intermediate states. 
TODO: Is this the best way to implement this?'''
def _find_last_obs(hydra : HydraAgent):
    i = -1
    while hydra.observations[i].intermediate_states is None:
        i=i-1
    return hydra.observations[i]