
import pytest

from os import path
import settings
from agent.planning.pddlplus_parser import PddlProblemParser, PddlDomainParser
from agent.planning.pddl_plus import PddlPlusState, PddlPlusGrounder, TimedAction, PddlPlusPlan
from agent.consistency.pddl_plus_simulator import PddlPlusSimulator
from agent.planning.planner import Planner
DATA_DIR = path.join(settings.ROOT_PATH, 'data')



''' Return domain, problem '''
@pytest.fixture(scope="module")
def get_problem_and_domain():
    pddl_file_name = path.join(DATA_DIR, "pddl_parser_test_domain.pddl")
    parser = PddlDomainParser()
    pddl_domain = parser.parse_pddl_domain(pddl_file_name)
    assert pddl_domain is not None, "PDDL+ domain object not parsed"

    pddl_file_name = path.join(DATA_DIR, "pddl_parser_test_problem.pddl")
    parser = PddlProblemParser()
    pddl_problem = parser.parse_pddl_problem(pddl_file_name)
    assert pddl_problem is not None, "PDDL+ problem object not parsed"

    return (pddl_problem, pddl_domain)


''' Return domain, problem, for experiment with planner '''
@pytest.fixture(scope="module")
def get_problem_and_domain_for_planner():
    pddl_file_name = path.join(DATA_DIR, "simulator_test_domain.pddl")
    parser = PddlDomainParser()
    pddl_domain = parser.parse_pddl_domain(pddl_file_name)
    assert pddl_domain is not None, "PDDL+ domain object not parsed"

    pddl_file_name = path.join(DATA_DIR, "simulator_test_problem.pddl")
    parser = PddlProblemParser()
    pddl_problem = parser.parse_pddl_problem(pddl_file_name)
    assert pddl_problem is not None, "PDDL+ problem object not parsed"

    return (pddl_problem, pddl_domain)



''' Return domain, problem, and flying process'''
@pytest.fixture(scope="module")
def get_process(get_problem_and_domain):

    # Get the flying process
    (pddl_problem, pddl_domain) = get_problem_and_domain
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
def get_action(get_problem_and_domain):
    (pddl_problem, pddl_domain) = get_problem_and_domain

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
def test_simulator_apply_effect(get_problem_and_domain, get_process):
    (pddl_problem, pddl_domain) = get_problem_and_domain
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
def test_check_preconditions(get_problem_and_domain, get_process):

    (pddl_problem, pddl_domain) = get_problem_and_domain
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
def test_simulate(get_problem_and_domain, get_action):
    (pddl_problem, pddl_domain) = get_problem_and_domain
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
def test_simulate_real_plan(get_problem_and_domain_for_planner):
    (pddl_problem, pddl_domain) = get_problem_and_domain_for_planner
    planner = Planner()
    action_list = planner.plan(pddl_problem, pddl_domain)

    assert action_list is not None
    assert len(action_list)>0

    ''' Conversion to PddlPlusPlan object '''
    plan = PddlPlusPlan(list())
    grounded_domain = PddlPlusGrounder().ground_domain(pddl_domain, pddl_problem) # Needed to identify plan action
    for (action_name, t) in action_list:
        # Get action
        for action in grounded_domain.actions:
            if action.name==action_name:
                plan.append(TimedAction(action, t))
                continue
            raise ValueError("Action %s not found in domain" % action_name)

    # Get the current state
    init_state = PddlPlusState(pddl_problem.init)

    simulator = PddlPlusSimulator()
    (current_state, t, trace) = simulator.simulate(init_state, plan, pddl_problem, pddl_domain, delta_t=0.1)
    assert trace[0][0]==current_state
    assert t == trace[-1][1]

    fluents_to_trace = [('x_bird', 'redbird_0'),('y_bird', 'redbird_0'),('x_pig', 'pig_10'),('y_pig', 'pig_10')]
    fluents_trace = simulator.trace_fluents(trace, fluents_to_trace)
    simulator.print_fluent_trace(fluents_to_trace, fluents_trace)

    #
    # prob_test = open("%s/sb_test_prob.pddl" % settings.PLANNING_DOCKER_PATH).read()
    #
    # pddl_problem_file = open("%s/sb_prob.pddl" % str(settings.PLANNING_DOCKER_PATH), "w+")
    # pddl_problem_file.write(prob_test)
    # pddl_problem_file.close()
    #
    # assert os.stat("%s/sb_prob.pddl" % settings.PLANNING_DOCKER_PATH).st_size > 0
    #
    # planner = pl.Planner()
    # actions = planner.get_plan_actions()
    #
    # assert os.stat("%s/docker_build_trace.txt" % settings.PLANNING_DOCKER_PATH).st_size > 0
    #
    # assert os.stat("%s/docker_plan_trace.txt" % settings.PLANNING_DOCKER_PATH).st_size > 0
    #
    # assert os.stat("%s/docker_build_trace.txt" % settings.VAL_DOCKER_PATH).st_size > 0
    #
    # assert os.stat("%s/docker_validation_trace.txt" % settings.VAL_DOCKER_PATH).st_size > 0
    #
    # assert len(actions) > 0