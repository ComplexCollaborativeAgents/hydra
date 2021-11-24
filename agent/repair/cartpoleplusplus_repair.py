from agent.consistency.sequence_consistency_estimator import SequenceConsistencyEstimator
from agent.planning.cartpoleplusplus_pddl_meta_model import *
from agent.repair.meta_model_repair import *

''' Checks consistency by considering the location of the Cartpole fluents '''
class CartpolePlusPlusConsistencyEstimator(MetaModelBasedConsistencyEstimator):
    def __init__(self, unique_prefix_size = 100,discount_factor=0.9, consistency_threshold = 0.01):
        self.unique_prefix_size=unique_prefix_size
        self.discount_factor = discount_factor
        self.consistency_threshold = consistency_threshold

    ''' Estimate consitency by considering the location of the birds in the observed state seq '''
    def estimate_consistency(self, simulation_trace: list, state_seq: list, delta_t: float = DEFAULT_DELTA_T):
        fluent_names = []
        fluent_names.append(('pos_x',))
        fluent_names.append(('pos_y',))
        fluent_names.append(('theta_x',))
        fluent_names.append(('theta_y',))

        consistency_checker = SequenceConsistencyEstimator(fluent_names, self.unique_prefix_size, self.discount_factor, self.consistency_threshold)
        return consistency_checker.estimate_consistency(simulation_trace, state_seq, delta_t)

''' Repairs Cartpole constants '''
class CartpolePlusPlusRepair(MetaModelRepair):
    def __init__(self, consistency_checker=CartpolePlusPlusConsistencyEstimator(),
                 consistency_threshold=settings.CP_CONSISTENCY_THRESHOLD):
        meta_model = CartPolePlusPlusMetaModel()
        self.fluents_to_repair = meta_model.repairable_constants
        self.repair_deltas = meta_model.repair_deltas
        self.meta_model_repair = GreedyBestFirstSearchMetaModelRepair(self.fluents_to_repair,
                                                                      consistency_checker,
                                                                      self.repair_deltas,
                                                                      consistency_threshold)

    ''' Repair the given domain and plan such that the given plan's expected outcome matches the observed outcome'''
    def repair(self, pddl_meta_model, observation, delta_t=1.0):
        return self.meta_model_repair.repair(pddl_meta_model, observation, delta_t)
