import pytest
from os import path
import settings
from agent.planning.pddlplus_parser import PddlProblemParser, PddlDomainParser
from agent.planning.pddl_plus import PddlPlusState, PddlPlusGrounder, TimedAction, PddlPlusPlan, PddlPlusDomain, PddlPlusProblem
from agent.consistency.pddl_plus_simulator import PddlPlusSimulator
from agent.planning.planner import Planner


DATA_DIR = path.join(settings.ROOT_PATH, 'data')


def __get_problem_and_domain(problem_file_name :str, domain_file_name: str):
    pddl_file_name = path.join(DATA_DIR, domain_file_name)
    parser = PddlDomainParser()
    pddl_domain = parser.parse_pddl_domain(pddl_file_name)
    assert pddl_domain is not None, "PDDL+ domain object not parsed"

    pddl_file_name = path.join(DATA_DIR, problem_file_name)
    parser = PddlProblemParser()
    pddl_problem = parser.parse_pddl_problem(pddl_file_name)
    assert pddl_problem is not None, "PDDL+ problem object not parsed"

    return (pddl_problem, pddl_domain)

''' Helper funciton for tests. It simulates the execution of a plan on the given problem, 
and asserts that the goal has been achieved in the simulation'''
def __validate_working_plan(problem: PddlPlusProblem, domain: PddlPlusDomain, plan_trace_file: str, delta_t : float=0.05):
    planner = Planner()
    grounded_domain = PddlPlusGrounder().ground_domain(domain, problem)  # Needed to identify plan action
    plan = planner.extract_plan_from_plan_trace(plan_trace_file, grounded_domain)

    assert plan is not None
    assert len(plan)>0

    # Get the current state
    init_state = PddlPlusState(problem.init)

    simulator = PddlPlusSimulator()
    (current_state, t, trace) = simulator.simulate(init_state, plan, problem, domain, delta_t)

    for goal_fluent in problem.goal:  # (pig_dead pig_28)
        assert tuple(goal_fluent) in current_state.boolean_fluents



''' Return domain, problem, and flying process'''
@pytest.fixture(scope="module")
def get_process():
    # Get the flying process
    (pddl_problem, pddl_domain) = __get_problem_and_domain("pddl_parser_test_problem.pddl","pddl_parser_test_domain.pddl")
    flying_process = None
    for process in pddl_domain.processes:
        if process.name=="flying":
            flying_process = process
            break
    assert flying_process is not None

    # Ground it
    binding = dict()
    binding["?b"] = "redBird_0"
    grounded_flying_process = PddlPlusGrounder().ground_world_change(process, binding)

    return grounded_flying_process

''' Return domain, problem, and action'''
@pytest.fixture(scope="module")
def get_action():
    (pddl_problem, pddl_domain) = __get_problem_and_domain("pddl_parser_test_problem.pddl","pddl_parser_test_domain.pddl")

    # Get the flying process
    twang_action = None
    for action in pddl_domain.actions:
        if action.name=="pa-twang":
            twang_action = action
            break
    assert twang_action is not None

    # Ground it
    binding = dict()
    binding["?b"] = "redBird_0"
    grounded_twang_action = PddlPlusGrounder().ground_world_change(twang_action, binding)

    return grounded_twang_action


'''
    Check if apply effects works
'''
def test_simulator_apply_effect(get_process):
    (pddl_problem, pddl_domain) = __get_problem_and_domain("pddl_parser_test_problem.pddl",
                                                           "pddl_parser_test_domain.pddl")
    grounded_flying_process = get_process

    simulator = PddlPlusSimulator()

    # Get the current state
    init_state = PddlPlusState(pddl_problem.init)

    # Apply only the move x effect
    increase_x_effect = None
    for effect in grounded_flying_process.effects:
        if effect[0]=="increase" and effect[1][0]=="y_bird":
            increase_y_effect = effect
    assert increase_y_effect is not None

    # Assert initial values are as expected
    assert float(init_state.numeric_fluents[("y_bird", "redBird_0")])==29
    init_state.numeric_fluents[("vy_bird", "redBird_0")] = 20

    simulator.apply_effects(init_state,[increase_y_effect],1) #This effectis: (increase(x_bird ?b)(*  # t (* 1.0 (vx_bird ?b))))

    assert float(init_state.numeric_fluents[("y_bird", "redBird_0")])==49


''' Checks if check preconditions hold '''
def test_check_preconditions(get_process):

    (pddl_problem, pddl_domain) = __get_problem_and_domain("pddl_parser_test_problem.pddl","pddl_parser_test_domain.pddl")
    grounded_flying_process = get_process

    simulator = PddlPlusSimulator()

    # Identify specific preconditions
    bird_released_precondition =None
    active_bird_precondition=None
    above_ground_precondition=None
    not_dead_precondition=None
    for precondition in grounded_flying_process.preconditions:
        if precondition[0]=="bird_released":
            bird_released_precondition = precondition
            continue
        if precondition[1][0]=="active_bird":
            active_bird_precondition=precondition
            continue
        if precondition[1][0]=="y_bird":
            above_ground_precondition=precondition
            continue
        if precondition[1][0]=="bird_dead":
            not_dead_precondition=precondition
            continue
        raise ValueError("Unexpected precondition %s" % precondition)

    assert bird_released_precondition is not None
    assert active_bird_precondition is not None
    assert above_ground_precondition is not None
    assert not_dead_precondition is not None


    # Get the current state
    init_state = PddlPlusState(pddl_problem.init)

    assert simulator.preconditions_hold(init_state, [bird_released_precondition])==False
    assert simulator.preconditions_hold(init_state, [active_bird_precondition]) == True
    assert simulator.preconditions_hold(init_state, [above_ground_precondition]) == True
    assert simulator.preconditions_hold(init_state, [not_dead_precondition]) == True
    assert simulator.preconditions_hold(init_state, [active_bird_precondition, above_ground_precondition, not_dead_precondition]) == True
    assert not simulator.preconditions_hold(init_state, grounded_flying_process.preconditions)


''' Tests the apply process functionality '''
def test_simulate(get_action):
    (pddl_problem, pddl_domain) = __get_problem_and_domain("pddl_parser_test_problem.pddl",
                                                           "pddl_parser_test_domain.pddl")
    twang_action = get_action

    simulator = PddlPlusSimulator()

    # Get the current state
    init_state = PddlPlusState(pddl_problem.init)
    delta_t = 0.05

    timed_action = TimedAction(twang_action, 20)
    plan = [timed_action]

    (current_state, t, trace) = simulator.simulate(init_state, plan, pddl_problem, pddl_domain, delta_t)
    assert t>4
    assert trace[0][0]==current_state
    assert t == trace[-1][1]

    for (state, t) in trace:
        print("\n ---- Time is %s ---- \n" % t)
        state.to_print()



''' Test the simulator by actually running a planner and simulating the trace of the plan it generates'''
def test_simulate_real_plan():
    (problem, domain) = __get_problem_and_domain("sb_prob_l1.pddl","sb_domain_l1.pddl")

    trace_file_name = path.join(DATA_DIR, "docker_plan_trace_l1.txt")
    __validate_working_plan(problem, domain, trace_file_name)
