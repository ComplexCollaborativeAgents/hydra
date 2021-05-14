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
    def __init__(self, allow_cascading_effects=False, apply_domain_refiner=True):
        self.context = dict()
        self.apply_domain_refiner = apply_domain_refiner
        super().__init__(allow_cascading_effects=allow_cascading_effects)

    ''' Simulate running the given plan from the start state '''
    def simulate(self, plan_to_simulate: PddlPlusPlan, problem: PddlPlusProblem, domain: PddlPlusDomain, delta_t:float, max_t:float = 1000, max_iterations: float = 1000):
        # Remove inapplicable events

        if self.apply_domain_refiner:
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

        result = self._internal_eval(element, state, delta_t)
        self.context[element_str] = result
        return result


    ''' Evaluates a given expression using the fluents in the given state, and delta f, if needed'''
    def _internal_eval(self, element, state: PddlPlusState, delta_t: float = -1):
        if isinstance(element, list):
            if is_op(element[0]): # If element is an operator
                assert len(element) == 3

                op_name = element[0]

                element1_str = str(element[1])
                if element1_str in self.context:
                    value1 = self.context[element1_str]
                else:
                    value1 = self._internal_eval(element[1], state, delta_t)
                    self.context[element1_str] = value1

                element2_str = str(element[2])
                if element2_str in self.context:
                    value2 = self.context[element2_str]
                else:
                    value2 = self._internal_eval(element[2], state, delta_t)
                    self.context[element2_str] = value2

                # assert(is_float(value1))
                # assert(is_float(value2))

                # Switch case
                if op_name == "+":
                    result = value1 + value2
                elif op_name == "-":
                    result = value1 - value2
                elif op_name == "*":
                    result = value1 * value2
                elif op_name == "/":
                    result = value1 / value2
                elif op_name == ">":
                    result = value1 > value2
                elif op_name == "<":
                    result = value1 < value2
                elif op_name == "<=":
                    result = value1 <= value2
                elif op_name == ">=":
                    result = value1 >= value2
                elif op_name == "=":
                    result = value1 == value2
                else:
                    result = eval("%s%s%s" % (str(value1), op_name, str(value2)))
                return result

            else: # Else element is a fluent value
                if len(element)==1: # This is a fluent
                    fluent_name = (element[0],) # Todo: understand why this hack is needed
                else:
                    fluent_name = tuple(element)
                if fluent_name in state.numeric_fluents:
                    return float(state.numeric_fluents[fluent_name])
                else:
                    return fluent_name in state.boolean_fluents
        else:
            if is_float(element):
                return float(element) # A constant
            elif element=="#t":
                if delta_t==-1:
                    raise ValueError("Delta t not set, but needed for evaluation")
                return delta_t
            else:
                return element
                # return float(state.numeric_fluents[element])  # A fluent


    ''' Checks if a set of preconditions hold in the given state'''
    def preconditions_hold(self, state, preconditions):
        for precondition in preconditions:
            if self.__precondition_hold(state, precondition)==False:
                return False
        return True

    ''' Checks if a single precondition hold in the given state'''
    def __precondition_hold(self, state : PddlPlusState, single_precondition):
        if single_precondition[0]=="not":
            return not self.__precondition_hold(state, single_precondition[1])
        if single_precondition[0]=="or":
            for precondition in single_precondition[:0:-1]:
                if self.__precondition_hold(state, precondition)==True:
                    return True
            return False
        return self._eval(single_precondition, state)