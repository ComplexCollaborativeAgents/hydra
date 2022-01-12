from agent.consistency.consistency_estimator import ConsistencyEstimator
from agent.consistency.pddl_plus_simulator import PddlPlusSimulator, InconsistentPlanError


class MetaModelBasedConsistencyEstimator(ConsistencyEstimator):
    PLAN_FAILED_CONSISTENCY_VALUE = 1000  # A constant representing the inconsistency value of a meta model in which the executed plan is inconsistent

    ''' Computes the consistency of a given observation w.r.t the given meta model using the given simulator '''
    def compute_consistency(self, observation, meta_model, simulator : PddlPlusSimulator, delta_t):
        try:
            expected_trace, plan = simulator.get_expected_trace(observation, meta_model, delta_t)
            observed_seq = observation.get_pddl_states_in_trace(meta_model)
            consistency = self.estimate_consistency(expected_trace, observed_seq, delta_t)
        except InconsistentPlanError: # Sometimes the repair makes the executed plan be inconsistent, e.g., its preconditions are not satisfied
            consistency = MetaModelBasedConsistencyEstimator.PLAN_FAILED_CONSISTENCY_VALUE
        return  consistency

class PolycraftConsistencyEstimator(MetaModelBasedConsistencyEstimator):
    def __init__(self, unique_prefix_size = 100,discount_factor=0.9, consistency_threshold = 0.01):
        self.unique_prefix_size=unique_prefix_size
        self.discount_factor = discount_factor
        self.consistency_threshold = consistency_threshold

    ''' Estimate consitency by considering the location of the birds in the observed state seq '''
    def estimate_consistency(self, simulation_trace: list, state_seq: list, delta_t: float = DEFAULT_DELTA_T):
        fluent_names = []
        fluent_names.append(('x',))
        fluent_names.append(('x_dot',))
        fluent_names.append(('theta',))
        fluent_names.append(('theta_dot',))

        consistency_checker = SequenceConsistencyEstimator(fluent_names, self.unique_prefix_size, self.discount_factor, self.consistency_threshold)
        return consistency_checker.estimate_consistency(simulation_trace, state_seq, delta_t)
