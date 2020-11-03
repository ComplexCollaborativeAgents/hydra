'''
    This module provides enhancements to our basic PDDL+ simulator
'''
from agent.consistency.pddl_plus_simulator import *


''' Analyzes the domain to refine it, identifying inapplicable events '''
class DomainRefiner:

    ''' Collect fluents used in a given state '''
    def find_state_fluents(state: PddlPlusState):
        fluents = set()
        fluents.update(state.boolean_fluents.keys())
        fluents.update(state.numeric_fluents.keys())
        return fluents

    ''' Collects the list of fluents used in the set of effects'''
    def add_fluents_from_effects(self, effects, fluents: set):
        for effect in effects:
            effect_type = effect[0]  # increase or decrease
            if effect_type in ("increase", "decrease", "assign"):
                # Numeric fluent
                fluent_name = tuple(effect[1])
                fluents.add(fluent_name)

                # Fluents for a formula
                self.add_fluents_from_formula(effect[2], fluents)

            else:  # Boolean effects
                if effect[0] == "not":  # remove a fact
                    fluent_name = tuple(effect[1])
                else:  # add a fact
                    fluent_name = tuple(effect)
                fluents.add(fluent_name)

    ''' Adds all the fluents used in the given formula'''
    def add_fluents_from_formula(self, formula, fluents: set):
        if isinstance(formula, list):  # This if prunes primitives
            if is_op(formula[0]):  # If element is an operator
                # assert len(formula) == 3
                self.add_fluents_from_formula(formula[1], fluents)
                self.add_fluents_from_formula(formula[2], fluents)
            else:  # Else element is a fluent value
                if len(formula) == 1:  # This is a fluent
                    fluent_name = (formula[0],)  # Todo: understand why this hack is needed
                else:
                    fluent_name = tuple(formula)
                fluents.add(fluent_name)

    ''' Adds all the fluents used in the given set of preconditions '''
    def add_fluents_in_precondition(self, precondition, fluents: set):
        if isinstance(precondition, str):
            if is_float(precondition)==False:
                fluents.add((precondition,))
        elif isinstance(precondition, list):
            if precondition[0] == "not":
                self.add_fluents_in_precondition(precondition[1], fluents)
            elif precondition[0] == "or":
                for clause in precondition[1:]:
                    self.add_fluents_in_precondition(clause, fluents)
            else:
                self.add_fluents_from_formula(precondition,fluents)
        return fluents

    ''' Finds all the effects that are not applicable in the current problem '''
    def find_inapplicable_events(self, grounded_domain: PddlPlusDomain, grounded_problem: PddlPlusProblem):
        initial_state = PddlPlusState(grounded_problem.init)

        world_changes = list()
        world_changes.extend(grounded_domain.actions)
        world_changes.extend(grounded_domain.events)
        world_changes.extend(grounded_domain.processes)

        fluents_in_effects = set()
        for world_change in world_changes:
            self.add_fluents_from_effects(world_change.effects, fluents_in_effects)

        sim = PddlPlusSimulator()
        fluents_in_precondition = set()
        inapplicable_events = set()
        for event in grounded_domain.events:
            for precondition in event.preconditions:
                fluents_in_precondition.clear()
                self.add_fluents_in_precondition(precondition, fluents_in_precondition)

                # if event's preconditions constants
                if len(set(fluents_in_effects).intersection(fluents_in_precondition)) == 0:
                    if sim.preconditions_hold(initial_state, [precondition]) == False:
                        inapplicable_events.add(event)
                        break

        return inapplicable_events

''' A simplistic, probably not complete and sound, simulator for PDDL+ processes
 TODO: Replace this with a call to VAL. '''
class RefinedPddlPlusSimulator(PddlPlusSimulator):
    ''' Simulate running the given plan from the start state '''
    def simulate(self, plan_to_simulate: PddlPlusPlan, problem: PddlPlusProblem, domain: PddlPlusDomain, delta_t:float, max_t:float = 1000, max_iterations: float = 1000):
        # Remove inapplicable events
        refiner = DomainRefiner()
        unused_events = refiner.find_inapplicable_events(domain, problem)
        for event in unused_events:
            domain.events.remove(event)

        return super().simulate(plan_to_simulate, problem, domain, delta_t, max_t, max_iterations)


''' A PDDL+ sim that caches calls to evaluate formulaes to gain efficiency  '''
class CachingPddlPlusSimulator(PddlPlusSimulator):
    def __init__(self):
        self.context = dict()

    ''' Simulate running the given plan from the start state '''
    def simulate(self, plan_to_simulate: PddlPlusPlan, problem: PddlPlusProblem, domain: PddlPlusDomain, delta_t:float, max_t:float = 1000, max_iterations: float = 1000):
        # Remove inapplicable events
        refiner = DomainRefiner()
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
