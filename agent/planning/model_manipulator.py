'''
This model manipulates a PDDL+ problem and domain files
'''
from agent.planning.pddl_plus import PddlPlusProblem, PddlPlusDomain
import agent.planning.pddl_plus as pddl_plus

''' General interface'''
class ProblemManipualtor():
    def apply_change(self, pddl_domain, pddl_problem):
        raise ValueError("Not implemented yet")


'''
Changes the given pddl+ problem file by adding delta to one of the numeric fluent in the initial state.
Note, this directly modifies the given pddl_problem object, it does not create a clone.  
'''
class ManipulateInitNumericFluent(ProblemManipualtor):
    def __init__(self, fluent_name, delta: float):
        self.fluent_to_change = fluent_name
        self.delta = delta

    def apply_change(self, pddl_domain, pddl_problem):
        fluent = pddl_plus.get_numeric_fluent(pddl_problem.init, self.fluent_to_change)
        old_value = pddl_plus.get_numeric_fluent_value(fluent)
        fluent[-1]= float(old_value)+self.delta


