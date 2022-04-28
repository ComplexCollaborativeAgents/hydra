from agent.consistency.sequence_consistency_estimator import SequenceConsistencyEstimator

from agent.repair.meta_model_repair import *

''' Checks consistency by considering the location of the Cartpole fluents '''
class CartpoleConsistencyEstimator(MetaModelBasedConsistencyEstimator):
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


class CartpoleRepair(GreedyBestFirstSearchMetaModelRepair):
    ''' Repairs Cartpole constants '''
    def __init__(self, consistency_checker=CartpoleConsistencyEstimator(),
                 consistency_threshold=settings.CP_CONSISTENCY_THRESHOLD):
        super().__init__(consistency_checker, consistency_threshold)

