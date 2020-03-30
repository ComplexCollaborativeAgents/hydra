import agent.planning.model_manipulator as model_manipulator

''' An abstract class intended to repair a given PDDL+ domain and problem until it matches the observed behavior '''
class ModelRepair():

    def repair(self, pddl_domain, pddl_problem):
        expected_obs = get_expected_observations(pddl_domain, pddl_problem) # From Val?
        real_obs = collect_real_observations(pddl_domain, pddl_problem) # From SB?
        while is_consistent(expected_obs, real_obs)==False:
            manipulator = self.choose_manipulator()
            manipulator.apply_change(pddl_domain, pddl_problem)

            expected_obs = get_expected_observations(pddl_domain, pddl_problem)

        return (pddl_domain, pddl_problem)

    def choose_manipulator(self):
        raise NotImplemented("Not yet")


'''
A basic model repair instance that changes a single fluent with fixed delta jumps, doing delta, -delta, 2 delta, -2 delta...
'''
class SingleNumericFluentRepair(ModelRepair):

    def __init__(self, fluent, delta):
        self.fluent_to_change = fluent
        self.delta = delta
        self.manipulator = model_manipulator.ManipulateInitNumericFluent(fluent, delta)

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


def get_expected_observations(pddl_domian, pddl_problem):
    raise NotImplementedError("Not yet")
def collect_real_observations(pddl_domain, pddl_problem):
    raise NotImplementedError("Not yet")
def is_consistent(obs1, obs2):
    raise NotImplementedError("Not yet")