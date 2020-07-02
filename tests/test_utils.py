import matplotlib.pyplot as plt

from agent.consistency.meta_model_repair import *
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

    for (action_name, time) in plan_actions:
        assert action_name.startswith("pa-twang")
        time = float(time)
        action_obj = grounded_domain.get_action(action_name)
        if action_obj is None:
            raise ValueError("Action name %s is not in the grounded domain" % action_name)

        timed_action = TimedAction(action_obj, time)
        plan.append(timed_action)
    return plan

''' Loads a plan for a given problem and domain, from a file. '''
def load_plan(plan_trace_file: str, pddl_problem: PddlPlusProblem, pddl_domain: PddlPlusDomain):
    planner = Planner()
    grounded_domain = PddlPlusGrounder().ground_domain(pddl_domain, pddl_problem)  # Needed to identify plan action
    pddl_plan = extract_plan_from_plan_trace(plan_trace_file, grounded_domain)
    return pddl_plan

''' Simulate the given action in the given state using the given meta model'''
def plot_expected_trace(meta_model: MetaModel,
                        state : ProcessedSBState,
                        time_action : list,
                        delta_t = 0.05):
    expected_trace = PddlPlusSimulator().simulate_observed_action(state, time_action, meta_model, delta_t)
    state_sequence = [timed_state[0] for timed_state in expected_trace]
    plot_state_sequence(state_sequence,meta_model.create_pddl_state(state))

''' Plot the given observation'''
def plot_observation(observation: ScienceBirdsObservation):
    meta_model = MetaModel()
    sb_state = observation.state
    pddl_state = meta_model.create_pddl_state(sb_state)
    obs_state_sequence = observation.get_trace(meta_model)
    plot_state_sequence(obs_state_sequence, pddl_state)

'''
Plotting a sequence of states, showing where the pigs, platforms, and birds are initially,
and showing the trajectory of the active bird. 
'''
def plot_state_sequence(state_seq : list, pddl_state: PddlPlusState):
    fig, ax = plt.subplots()
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
    x_active_bird = [state[("x_bird", active_bird)] for state in state_seq]
    y_active_bird = [state[("y_bird", active_bird)] for state in state_seq]
    obs_points = set(zip(x_active_bird, y_active_bird))
    x_active_bird = [state[0] for state in obs_points]
    y_active_bird = [state[1] for state in obs_points]
    # Set plot area to be a square, so that proprtions are right.
    (left, right_x) = plt.xlim()
    (left, right_y) = plt.ylim()
    max_axis = max(right_x, right_y)
    plt.xlim((0, max_axis))
    plt.ylim((0, max_axis))
    ax.plot(x_active_bird, y_active_bird, marker="x", markersize=8, linestyle="")
    plt.show()


''' A planner that executes a predefined plan '''
class PlannerStub(Planner):
    ''' plan is assumed to be list of timed actions. '''
    def __init__(self, plan, meta_model : MetaModel = MetaModel()):
        super(PlannerStub, self).__init__(meta_model)
        self.plan = list(plan)

    ''' @Override superclass '''
    def make_plan(self, state: ProcessedSBState, prob_complexity=0, delta_t = 1.0):
        action = self.plan.pop(0)
        return [action]


