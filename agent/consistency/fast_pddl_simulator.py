'''
    This module provides enhancements to our basic PDDL+ simulator
'''
from agent.consistency.pddl_plus_simulator import *
from agent.planning.domain_analyzer import DomainAnalyzer

''' A simplistic, probably not complete and sound, simulator for PDDL+ processes
 TODO: Replace this with a call to VAL. '''
class RefinedPddlPlusSimulator(PddlPlusSimulator):
    ''' Simulate running the given plan from the start state '''
    def simulate(self, plan_to_simulate: PddlPlusPlan, problem: PddlPlusProblem, domain: PddlPlusDomain, delta_t:float, max_t:float = 1000, max_iterations: float = 1000):
        # Remove inapplicable events
        unused_events = self.find_inapplicable_events(domain, problem)
        for event in unused_events:
            domain.events.remove(event)

        return super().simulate(plan_to_simulate, problem, domain, delta_t, max_t, max_iterations)

    ''' Finds all the effects that are not applicable in the current problem '''
    def find_inapplicable_events(self, grounded_domain: PddlPlusDomain, grounded_problem: PddlPlusProblem):
        domain_analyzer = DomainAnalyzer()
        initial_state = PddlPlusState(grounded_problem.init)

        world_changes = list()
        world_changes.extend(grounded_domain.actions)
        world_changes.extend(grounded_domain.events)
        world_changes.extend(grounded_domain.processes)

        fluents_in_effects = set()
        for world_change in world_changes:
            domain_analyzer.add_fluents_from_effects(world_change.effects, fluents_in_effects)

        sim = PddlPlusSimulator()
        fluents_in_precondition = set()
        inapplicable_events = set()
        for event in grounded_domain.events:
            for precondition in event.preconditions:
                fluents_in_precondition.clear()
                domain_analyzer.add_fluents_in_precondition(precondition, fluents_in_precondition)

                # if event's preconditions constants
                if len(set(fluents_in_effects).intersection(fluents_in_precondition)) == 0:
                    if sim.preconditions_hold(initial_state, [precondition]) == False:
                        inapplicable_events.add(event)
                        break

        return inapplicable_events


''' A PDDL+ sim that caches calls to evaluate formulaes to gain efficiency  '''
class CachingPddlPlusSimulator(PddlPlusSimulator):
    def __init__(self):
        self.context = dict()

    ''' Simulate running the given plan from the start state '''
    def simulate(self, plan_to_simulate: PddlPlusPlan, problem: PddlPlusProblem, domain: PddlPlusDomain, delta_t:float, max_t:float = 1000, max_iterations: float = 1000):
        # Remove inapplicable events
        refiner = DomainAnalyzer()
        unused_events = refiner.find_inapplicable_events(domain, problem)
        for event in unused_events:
            domain.events.remove(event)

        return super().simulate(plan_to_simulate, problem, domain, delta_t, max_t, max_iterations)

    def _sim_step(self, current_state, t):
        self.context.clear()  # New t value may change the cached values TODO: Smarter caching
        return super()._sim_step(current_state, t)

    ''' Apply the specified effect ont he given state '''
    def apply_effects(self, state, effects, delta_t=-1):
        self.context.clear() # Effects may change the cached values TODO: Smarter caching
        super().apply_effects(state, effects,delta_t)

    ''' Evaluates a given expression using the fluents in the given state, and delta f, if needed'''
    def _eval(self, element, state: PddlPlusState, delta_t: float = -1):
        element_str = str(element)
        if element_str in self.context:
            return self.context[element_str]
        else:
            result = super()._eval(element, state, delta_t)
            self.context[element_str] = result
            return result
