import matplotlib.pyplot as plt

from agent.consistency.meta_model_repair import *
from agent.consistency.observation import ScienceBirdsObservation
from agent.planning.planner import *
from agent.planning.planner import Planner
import matplotlib.patches as patches

Y_BIRD_FLUENT = ('y_bird', 'redbird_0')
X_BIRD_FLUENT = ('x_bird', 'redbird_0')

''' Helper function: simulate the given plan, on the given problem and domain.  '''
def simulate_plan_trace(plan: PddlPlusPlan, problem:PddlPlusProblem, domain: PddlPlusDomain, delta_t:float = 0.05):
    simulator = PddlPlusSimulator()
    (_, _, trace) =  simulator.simulate(plan, problem, domain, delta_t)
    return trace

''' Helper function: returns a PDDL+ problem and domain objects'''
def load_problem_and_domain(problem_file_name :str, domain_file_name: str):
    parser = PddlDomainParser()
    pddl_domain = parser.parse_pddl_domain(domain_file_name)
    assert pddl_domain is not None, "PDDL+ domain object not parsed"

    parser = PddlProblemParser()
    pddl_problem = parser.parse_pddl_problem(problem_file_name)
    assert pddl_problem is not None, "PDDL+ problem object not parsed"

    return (pddl_problem, pddl_domain)

''' Extracts a PddlPlusPlan object from a plan trace. '''
def extract_plan_from_plan_trace(plan_trace_file, grounded_domain :PddlPlusDomain):
    planner = Planner()
    plan_actions = planner.extract_actions_from_plan_trace(plan_trace_file)
    plan = PddlPlusPlan()

    for (action_name, angle, time) in plan_actions:
        assert action_name.startswith("pa-twang")
        time = float(time)
        timed_action = TimedAction(action_name, time)
        plan.append(timed_action)
    return plan

''' Loads a plan for a given problem and domain, from a file. '''
def load_plan(plan_trace_file: str, pddl_problem: PddlPlusProblem, pddl_domain: PddlPlusDomain):
    planner = Planner()
    grounded_domain = PddlPlusGrounder().ground_domain(pddl_domain, pddl_problem)  # Needed to identify plan action
    pddl_plan = extract_plan_from_plan_trace(plan_trace_file, grounded_domain)
    return pddl_plan

''' Simulate the given action in the given state using the given meta model'''
def plot_expected_trace_for_obs(meta_model: MetaModel,
                                observation: ScienceBirdsObservation,
                                delta_t = 0.05,
                                ax=None):
    # Repair angle
    expected_trace = PddlPlusSimulator().simulate_observed_action(observation.state, observation.action, meta_model, delta_t)
    state_sequence = [timed_state[0] for timed_state in expected_trace]
    return plot_state_sequence(state_sequence, meta_model.create_pddl_state(observation.state),ax)

''' Plot the given observation'''
def plot_observation(observation: ScienceBirdsObservation, ax=None, marker="o"):
    meta_model = MetaModel()
    sb_state = observation.state
    pddl_state = meta_model.create_pddl_state(sb_state)
    obs_state_sequence = observation.get_trace(meta_model)
    return plot_state_sequence(obs_state_sequence, pddl_state, ax, marker)

'''
Plotting a sequence of states, showing where the pigs, platforms, and birds are initially,
and showing the trajectory of the active bird. 
'''
def plot_state_sequence(state_seq : list, pddl_state: PddlPlusState, ax= None, marker="x"):
    if ax is None:
        _, ax = plt.subplots()
    # plot pigs
    pigs = pddl_state.get_pigs()
    x_pigs = [pddl_state[("x_pig", pig)] for pig in pigs]
    y_pigs = [pddl_state[("y_pig", pig)] for pig in pigs]
    ax.plot(x_pigs, y_pigs, marker="$pig$", markersize=19, linestyle="")
    # plot birds
    birds = pddl_state.get_birds()
    x_birds = [pddl_state[("x_bird", bird)] for bird in birds]
    y_birds = [pddl_state[("y_bird", bird)] for bird in birds]
    ax.plot(x_birds, y_birds, marker="$bird$", markersize=19, linestyle="")
    platforms = pddl_state.get_platforms()
    for platform in platforms:
        x = pddl_state[("x_platform", platform)]
        y = pddl_state[("y_platform", platform)]
        width = pddl_state[("platform_width", platform)]
        height = pddl_state[("platform_height", platform)]
        x = x - width / 2
        y = y - height / 2
        rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    # plot active bird trajectory
    active_bird = pddl_state.get_active_bird()
    obs_points = [] # Remove
    last_point = None
    x_bird_fluent = ("x_bird", active_bird)
    y_bird_fluent = ("y_bird", active_bird)
    for state in state_seq:
        if (x_bird_fluent not in state) or (y_bird_fluent not in state):
            continue

        new_point = (state[x_bird_fluent], state[y_bird_fluent])
        if last_point is not None and last_point==new_point:
            continue
        obs_points.append(new_point)
        last_point=new_point

    x_active_bird = [state[0] for state in obs_points]
    y_active_bird = [state[1] for state in obs_points]
    ax.plot(x_active_bird, y_active_bird, marker=marker, markersize=8, linestyle="")

    # Set plot area to be a square, so that proportions are right.
    (left, right_x) = plt.xlim()
    (left, right_y) = plt.ylim()
    max_axis = max(right_x, right_y)
    plt.xlim((0, max_axis))
    plt.ylim((0, max_axis))

    plt.show()
    return ax

