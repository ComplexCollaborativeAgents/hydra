import matplotlib.pyplot as plt

from agent.consistency.meta_model_repair import *
from agent.planning.planner import *
from agent.planning.planner import MetaModelBasedPlanner
from worlds.science_birds import SBState

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

''' Loads a plan for a given problem and domain, from a file. '''
def load_plan(plan_trace_file: str, pddl_problem: PddlPlusProblem, pddl_domain: PddlPlusDomain):
    planner = Planner()
    grounded_domain = PddlPlusGrounder().ground_domain(pddl_domain, pddl_problem)  # Needed to identify plan action
    pddl_plan = planner.extract_plan_from_plan_trace(plan_trace_file, grounded_domain)
    return pddl_plan

''' Simulate the given action in the given state using the given meta model'''
def plot_expected_trace(meta_model: MetaModel,
                        state : SBState,
                        time_action : list,
                        marker: str = "o",
                        fluent_x = ('x_bird', 'redbird_0'),
                        fluent_y = ('y_bird', 'redbird_0'), delta_t = 0.05):
    expected_trace = PddlPlusSimulator().simulate_observed_action(state, time_action, meta_model, delta_t)
    plt.plot([timed_state[0][fluent_x] for timed_state in expected_trace],
             [timed_state[0][fluent_y] for timed_state in expected_trace],marker=marker)

''' Plot the given observation'''
def plot_observation(observation: ScienceBirdsObservation):
    meta_model = MetaModel()
    sb_state = observation.state
    pddl_state = meta_model.create_pddl_state(sb_state)
    fig, ax = plt.subplots()

    # plot pigs
    pigs = pddl_state.get_pigs()
    x_pigs = [pddl_state[("x_pig",pig)] for pig in pigs]
    y_pigs = [pddl_state[("y_pig", pig)] for pig in pigs]
    ax.plot(x_pigs, y_pigs, marker="$pig$", markersize=19, linestyle="")

    # plot birds
    birds = pddl_state.get_birds()
    x_birds = [pddl_state[("x_bird",bird)] for bird in birds]
    y_birds = [pddl_state[("y_bird", bird)] for bird in birds]
    ax.plot(x_birds, y_birds, marker="$bird$", markersize=19, linestyle="")

    # plot active bird trajectory
    active_bird = pddl_state.get_active_bird()
    observed_seq = observation.get_trace(meta_model)

    x_active_bird= [state[("x_bird",active_bird)] for state in observed_seq]
    y_active_bird= [state[("y_bird",active_bird)] for state in observed_seq]
    obs_points = set(zip(x_active_bird,y_active_bird))
    x_active_bird = [state[0] for state in obs_points]
    y_active_bird = [state[1] for state in obs_points]
    ax.plot(x_active_bird, y_active_bird, marker="x", markersize=8, linestyle="")

    plt.show()


''' A planner that executes a predefined plan '''
class PlannerStub(MetaModelBasedPlanner):
    ''' plan is assumed to be list of timed actions. '''
    def __init__(self, plan, meta_model : MetaModel = MetaModel()):
        super(PlannerStub, self).__init__(meta_model)
        self.plan = list(plan)

    ''' @Override superclass '''
    def make_plan(self, state: SBState, prob_complexity=0, delta_t = 1.0):
        action = self.plan.pop(0)
        return [action]


