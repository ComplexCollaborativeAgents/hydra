import agent.planning.model_manipulator as model_manipulator
from agent.consistency.pddl_plus_simulator import PddlPlusSimulator
from agent.consistency.consistency_estimator import *

''' An abstract class intended to repair a given PDDL+ domain and problem until it matches the observed behavior '''
class ModelRepair():

    ''' Repair the given domain and plan such that the given plan's expected outcome matches the observed outcome'''
    def repair(self, pddl_domain : PddlPlusDomain, pddl_problem : PddlPlusProblem, pddl_plan : PddlPlusPlan,
               observed_states : list, delta_t = 0.05):
        simulator = PddlPlusSimulator()
        (_,_,expected_obs) = simulator.simulate(pddl_plan, pddl_problem, pddl_domain, delta_t)
        manipulator_itr = self.choose_manipulator()
        while self.is_consistent(expected_obs, observed_states)==False:
            manipulator = next(manipulator_itr)
            manipulator.apply_change(pddl_domain, pddl_problem)

            (_,_,expected_obs) = simulator.simulate(pddl_plan, pddl_problem, pddl_domain,delta_t)

        return (pddl_domain, pddl_problem)

    ''' The first parameter is a list of (state, time) pairs, the second is just a list of states. 
     Checks if they can be aligned. '''
    def is_consistent(self, timed_state_seq: list, state_seq: list):
        raise NotImplementedError("Not yet")

    def choose_manipulator(self):
        raise NotImplemented("Not yet")


'''
A basic model repair instance that changes a single fluent with fixed delta jumps, doing delta, -delta, 2 delta, -2 delta...
'''
class SingleNumericFluentRepair(ModelRepair):

    def __init__(self, fluent_to_repair, fluent_for_consistency_check, delta):
        self.fluent_to_change = fluent_to_repair
        self.delta = delta
        self.manipulator = model_manipulator.ManipulateInitNumericFluent(fluent_to_repair, delta)
        self.consistency_checker = SingleNumericFluentConsistencyEstimator(fluent_for_consistency_check)

    '''
    Simple brute force in the space of delta factors. It goes like this: +delta, -delta, +2delta, -2delta
    '''
    def choose_manipulator(self):
        delta_sum = self.delta
        delta_sign = +1
        while True:
            yield self.manipulator

            delta_sign=delta_sign*-1
            delta_sum = delta_sum+self.delta

            self.manipulator.delta = delta_sum*delta_sign

    ''' The first parameter is a list of (state, time) pairs, the second is just a list of states '''
    def is_consistent(self, timed_state_seq: list, state_seq: list, consistency_threshold = 0.05):
        consistency_value = self.consistency_checker.estimate_consistency(timed_state_seq, state_seq)
        if consistency_value<consistency_threshold:
            return True
        else:
            return False
