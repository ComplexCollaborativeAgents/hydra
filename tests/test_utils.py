from agent.consistency.consistency_estimator import ScienceBirdsObservation
from agent.consistency.meta_model_repair import *
from agent.perception.perception import Perception
from agent.planning.pddl_meta_model import MetaModel
from agent.planning.planner import *
from agent.perception.perception import *
import matplotlib.pyplot as plt

from agent.planning.planner import MetaModelBasedPlanner
from worlds.science_birds import SBState

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

