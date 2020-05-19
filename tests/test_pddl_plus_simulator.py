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

GRAVITY_BAD_OBS = path.join(TRACE_DIR,"g_250_observed.p")
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

''' Helper funciton for tests. It simulates the execution of a plan on the given problem, 
and asserts that the goal has been achieved in the simulation'''
def _validate_working_plan(problem: PddlPlusProblem, domain: PddlPlusDomain, plan_trace_file: str, delta_t : float=0.05):
    planner = Planner()
    grounded_domain = PddlPlusGrounder().ground_domain(domain, problem)  # Needed to identify plan action
    plan = planner.extract_plan_from_plan_trace(plan_trace_file, grounded_domain)

    assert plan is not None
    assert len(plan)>0

    # Get the current state
    simulator = PddlPlusSimulator()
    (current_state, t, trace) = simulator.simulate(plan, problem, domain, delta_t)

    for goal_fluent in problem.goal:  # (pig_dead pig_28)
        assert tuple(goal_fluent) in current_state.boolean_fluents

''' Tests the simulator's behavior when setting the gravity to be 250 '''
@pytest.mark.skipif(True, reason="Need a new observation object")
def test_gravity_250():
    # Load observation as created by the _create_gravity_250_observation() function
    our_observation = pickle.load(open(GRAVITY_BAD_OBS, "rb"))
    meta_model = MetaModel()
    problem = meta_model.create_pddl_problem(our_observation.state)
    domain = meta_model.create_pddl_domain(our_observation.state)
    plan = test_utils.load_plan(PLAN_LEVEL_01_FILE, problem, domain)
    expected_trace_ok = test_utils.simulate_plan_on_observed_state(plan, our_observation,meta_model)

    # Inject fault and simulate
    meta_model.constant_numeric_fluents['gravity'] = 250.0
    problem = meta_model.create_pddl_problem(our_observation.state)
    domain = meta_model.create_pddl_domain(our_observation.state)
    plan = test_utils.load_plan(PLAN_LEVEL_01_FILE, problem, domain)
    expected_trace_faulty = test_utils.simulate_plan_on_observed_state(plan, our_observation,meta_model)
    obs_sequence = our_observation.get_trace(meta_model)

    # Plot each
    Y_BIRD_FLUENT = ('y_bird', 'redbird_0')
    X_BIRD_FLUENT = ('x_bird', 'redbird_0')

    expected_x_values = []
    expected_y_values = []
    for (state,_,_) in expected_trace_ok:
        if state[X_BIRD_FLUENT] and state[Y_BIRD_FLUENT]:
            expected_x_values.append(state[X_BIRD_FLUENT])
            expected_y_values.append(state[Y_BIRD_FLUENT])
    expected_x_values_bad = []
    expected_y_values_bad = []
    for (state,_,_) in expected_trace_faulty:
        if state[X_BIRD_FLUENT] and state[Y_BIRD_FLUENT]:
            expected_x_values_bad.append(state[X_BIRD_FLUENT])
            expected_y_values_bad.append(state[Y_BIRD_FLUENT])
    observed_x_values = []
    observed_y_values = []
    for state in obs_sequence:
        if state[X_BIRD_FLUENT] and state[Y_BIRD_FLUENT]:
            observed_x_values.append(state[X_BIRD_FLUENT])
            observed_y_values.append(state[Y_BIRD_FLUENT])
    plt.plot(expected_x_values,expected_y_values,'r--',
             expected_x_values_bad, expected_y_values_bad,'bs',
             observed_x_values, observed_y_values,'go')
    plt.show()

    print("Ok")


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
    timed_action = TimedAction(twang_action, 20)
    plan = [timed_action]

    (current_state, t, trace) = simulator.simulate(plan, pddl_problem, pddl_domain, delta_t)
    assert t>4
    assert trace[-1][0]==current_state
    assert t == trace[-1][1]



''' TODO: REMOVE THIS TEST, JUST FOR DEBUG!!! Tests simulator in a simple 2 pig level '''
def test_simulator_2_pig_level():
    problem_file = path.join(DATA_DIR, "science_birds", "tests", "bad_angle_rate_problem.pddl")
    domain_file = path.join(DATA_DIR,  "science_birds", "tests", "bad_angle_rate_domain.pddl")
    (pddl_problem, pddl_domain) = test_utils.load_problem_and_domain(problem_file,
                                                                     domain_file)
    twang_action = _get_twang_action()

    # Get the flying process
    twang_action = pddl_domain.get_action("pa-twang")
    # Ground it
    binding = dict()
    binding["?b"] = "redbird_0"
    twang_action = PddlPlusGrounder().ground_world_change(twang_action, binding)


    simulator = PddlPlusSimulator()

    # Get the current state
    delta_t = 0.05
    timed_action = TimedAction(twang_action, 1)
    plan = [timed_action]
    (current_state, t, trace) = simulator.simulate(plan, pddl_problem, pddl_domain, delta_t)
    expected_state_seq = [state[0] for state in trace]
    red_bird_x_values = [state[('x_bird', 'redbird_0')] for state in expected_state_seq]
    red_bird_y_values = [state[('y_bird', 'redbird_0')] for state in expected_state_seq]
    plt.plot(red_bird_x_values, red_bird_y_values, 'bs')
    plt.show()
    for start_at in [1.5,2,2.5,3,3.5]:
        timed_action = TimedAction(twang_action, start_at)
        plan = [timed_action]

        (current_state, t, trace) = simulator.simulate(plan, pddl_problem, pddl_domain, delta_t)

        expected_state_seq  = [state[0] for state in trace]
        red_bird_x_values = [state[('x_bird', 'redbird_0')] for state in expected_state_seq]
        red_bird_y_values = [state[('y_bird', 'redbird_0')] for state in expected_state_seq]
        plt.plot(red_bird_x_values,red_bird_y_values,'d')
        plt.show()
        print(3)



''' Test the simulator by actually running a planner and simulating the trace of the plan it generates'''
def test_simulate_real_plan():
    problem_file = path.join(DATA_DIR, "sb_prob_l1.pddl")
    domain_file = path.join(DATA_DIR, "sb_domain_l1.pddl")

    (problem, domain) = test_utils.load_problem_and_domain(problem_file,domain_file)

    trace_file_name = path.join(DATA_DIR, "docker_plan_trace_l1.txt")
    _validate_working_plan(problem, domain, trace_file_name)
