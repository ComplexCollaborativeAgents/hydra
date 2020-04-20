import pytest
import os.path as path
import settings
from agent.planning.pddlplus_parser import PddlProblemParser, PddlProblemExporter, PddlDomainParser, PddlDomainExporter
from agent.planning.model_manipulator import ManipulateInitNumericFluent
from agent.planning.planner import Planner
from agent.consistency.model_repair import *
import agent.planning.pddl_plus as pddl_plus


DATA_DIR = path.join(settings.ROOT_PATH, 'data')
TRACE_DIR = path.join(DATA_DIR, 'science_birds', 'serialized_levels', 'level-01')
PRECISION = 0.0001

GRAVITY = ["gravity"]
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

''' Test changing a single numeric fluent, and the ability of model repair to repair it'''
def test_single_numeric_repair(get_plan_problem_domain):
    DELTA_T = 0.05

    # Get expected timed state sequence according to the model (plan, problem, domain)
    (pddl_plan, pddl_problem, pddl_domain) = get_plan_problem_domain
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



