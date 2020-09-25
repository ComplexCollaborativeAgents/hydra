from agent.consistency.sequence_consistency_estimator import SequenceConsistencyEstimator
from agent.consistency.consistency_estimator import *

from agent.consistency.meta_model_repair import *

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

''' Repairs Cartpole constants '''
class CartpoleRepair(MetaModelRepair):
    def __init__(self, consistency_checker, repair_deltas, consistency_threshold):
        self.fluents_to_repair = ["m_cart", "friction_cart", 'l_pole', 'm_pole', 'friction_pole', 'F', 'inertia','gravity']
        self.repair_deltas = [0.5, 0.5, 0.25, 0.1, 0.2, 1.0, 1.0]
        self.meta_model_repair = GreedyBestFirstSearchMetaModelRepair(self.fluents_to_repair,
                                                                      consistency_checker,
                                                                      repair_deltas,
                                                                      consistency_threshold)

        ''' Repair the given domain and plan such that the given plan's expected outcome matches the observed outcome'''
        def repair(self, pddl_meta_model, observation, delta_t=1.0):
            return self.meta_model_repair.repair(pddl_meta_model, observation, delta_t)