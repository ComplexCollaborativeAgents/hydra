'''
This model manipulates a PDDL+ problem and domain files
'''
from agent.planning.pddlplus_parser import PddlPlusDomain, PddlPlusProblem
import agent.planning.pddl_plus as pddl_plus


class PddlProblemManipulator():
    '''
    Changes the given pddl+ problem file by adding delta to one of the numeric fluent in the initial state.
    Note, this directly modifies the given pddl_problem object, it does not create a clone.  
    '''
    def change_init_numeric_fluent(self, pddl_domain : PddlPlusDomain, pddl_problem : PddlPlusProblem, fluent_to_change: str, delta: float):
        fluent = pddl_plus.get_numeric_fluent(pddl_problem.init, fluent_to_change)
        old_value = pddl_plus.get_numeric_fluent_value(fluent)
        fluent[-1]= float(old_value)+delta


