
from agent.consistency.meta_model_repair import *
from agent.planning.planner import *
from agent.perception.perception import *
import matplotlib.pyplot as plt

Y_BIRD_FLUENT = ('y_bird', 'redbird_0')
X_BIRD_FLUENT = ('x_bird', 'redbird_0')

''' Helper function: simulate the given plan, on the given problem and domain.  '''
def simulate_plan_trace(plan: PddlPlusPlan, problem:PddlPlusProblem, domain: PddlPlusDomain, delta_t:float = 0.05):
    simulator = PddlPlusSimulator()
    (_, _, trace) =  simulator.simulate(plan, problem, domain, delta_t)
    return trace

''' Simulate the outcome of performing an observed action in the observed state '''
def simulate_plan_on_observed_state(plan: PddlPlusPlan,
                             our_observation: ScienceBirdsObservation,
                             meta_model :MetaModel, delta_t = 0.05):
    assert our_observation.action is not None
    assert our_observation.state is not None

    pddl_problem = meta_model.create_pddl_problem(our_observation.state)
    pddl_domain = meta_model.create_pddl_domain(our_observation.state)
    plan_prefix = []
    for timed_action in plan:
        plan_prefix.append(timed_action)
        if timed_action.action.name == our_observation.action[0]:
            break
    return simulate_plan_trace(plan_prefix, pddl_problem, pddl_domain, delta_t)

''' Create a sequence of PDDL states from the observed sequence of intermediate SBStates '''
def extract_intermediate_states(our_observation: ScienceBirdsObservation, meta_model : MetaModel = MetaModel()):
    observed_state_seq = []
    perception = Perception()
    for intermediate_state in our_observation.intermediate_states:
        if isinstance(intermediate_state.objects, list):
            intermediate_state = perception.process_sb_state(intermediate_state)
        observed_state_seq.append(meta_model.create_pddl_state(intermediate_state))
    return observed_state_seq

''' Plots the values of the given pair of fluents of the two series. Useful to compare state sequences '''
def plot_bird_xy_series(serie1, serie2, fluent_names = [X_BIRD_FLUENT,Y_BIRD_FLUENT]):
    # Plot each
    expected_x_values = []
    expected_y_values = []
    for state in serie1:
        if state[X_BIRD_FLUENT] and state[Y_BIRD_FLUENT]:
            expected_x_values.append(state[X_BIRD_FLUENT])
            expected_y_values.append(state[Y_BIRD_FLUENT])
    observed_x_values = []
    observed_y_values = []
    for state in serie2:
        if state[X_BIRD_FLUENT] and state[Y_BIRD_FLUENT]:
            observed_x_values.append(state[X_BIRD_FLUENT])
            observed_y_values.append(state[Y_BIRD_FLUENT])
    plt.plot(expected_x_values,expected_y_values,'r--',observed_x_values,observed_y_values,'bs')
    plt.show()



''' Helper function: returns a PDDL+ problem and domain objects'''
def load_problem_and_domain(problem_file_name :str, domain_file_name: str):
    parser = PddlDomainParser()
    pddl_domain = parser.parse_pddl_domain(domain_file_name)
    assert pddl_domain is not None, "PDDL+ domain object not parsed"

    parser = PddlProblemParser()
    pddl_problem = parser.parse_pddl_problem(problem_file_name)
    assert pddl_problem is not None, "PDDL+ problem object not parsed"

    return (pddl_problem, pddl_domain)

''' Loads a plan for a given problem and domain, from a file. '''
def load_plan(plan_trace_file: str, pddl_problem: PddlPlusProblem, pddl_domain: PddlPlusDomain):
    planner = Planner()
    grounded_domain = PddlPlusGrounder().ground_domain(pddl_domain, pddl_problem)  # Needed to identify plan action
    pddl_plan = planner.extract_plan_from_plan_trace(plan_trace_file, grounded_domain)
    return pddl_plan