from os import path

import pytest
import matplotlib.pyplot as plt
import pickle
from agent.consistency.meta_model_repair import *
from agent.planning.planner import Planner
import tests.test_utils as test_utils
import matplotlib.pyplot as plt


DATA_DIR = path.join(settings.ROOT_PATH, 'data')
TRACE_DIR = path.join(DATA_DIR, 'science_birds', 'serialized_levels', 'level-01')
PROBLEM_TEST_FILE = path.join(DATA_DIR, "pddl_parser_test_problem.pddl")
DOMAIN_TEST_FILE = path.join(DATA_DIR, "pddl_parser_test_domain.pddl")

PLAN_LEVEL_01_FILE = path.join(TRACE_DIR,'docker_plan_trace.txt')


''' Helper function: returns the flying process'''
def _get_flying_process():
    # Get the flying process
    (pddl_problem, pddl_domain) = test_utils.load_problem_and_domain(PROBLEM_TEST_FILE,
                                                                     DOMAIN_TEST_FILE)
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

''' Helper function: returns the twang action '''
def _get_twang_action():
    (pddl_problem, pddl_domain) = test_utils.load_problem_and_domain(PROBLEM_TEST_FILE,
                                                                     DOMAIN_TEST_FILE)
    # Get the flying process
    twang_action = pddl_domain.get_action("pa-twang")
    assert twang_action is not None

    # Ground it
    binding = dict()
    binding["?b"] = "redBird_0"
    grounded_twang_action = PddlPlusGrounder().ground_world_change(twang_action, binding)

    return grounded_twang_action



''' 
    Check if apply effects works
'''
def test_simulator_apply_effect():
    (pddl_problem, pddl_domain) = test_utils.load_problem_and_domain(PROBLEM_TEST_FILE,
                                                                     DOMAIN_TEST_FILE)
    grounded_flying_process = _get_flying_process()

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
def test_check_preconditions():

    (pddl_problem, pddl_domain) = test_utils.load_problem_and_domain(PROBLEM_TEST_FILE,
                                                                     DOMAIN_TEST_FILE)
    grounded_flying_process = _get_flying_process()

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


''' Tests the simulator '''
def test_simulate():
    (pddl_problem, pddl_domain) = test_utils.load_problem_and_domain(PROBLEM_TEST_FILE,
                                                                     DOMAIN_TEST_FILE)
    twang_action = _get_twang_action()
    simulator = PddlPlusSimulator()

    # Get the current state
    delta_t = 0.05
    timed_action = TimedAction(twang_action.name, 20)
    plan = [timed_action]

    (current_state, t, trace) = simulator.simulate(plan, pddl_problem, pddl_domain, delta_t)
    assert t>4
    assert trace[-1][0]==current_state
    assert t == trace[-1][1]
