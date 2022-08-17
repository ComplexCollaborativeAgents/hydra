from agent.repair.meta_model_repair import *


class CartpolePlusPlusConsistencyEstimator(DomainConsistency):
    def __init__(self):
        super(CartpolePlusPlusConsistencyEstimator, self).__init__([CartPlusPlusLocationConsistency()])


class CartPlusPlusLocationConsistency(AspectConsistency):
    """ Checks consistency by considering the location of the Cartpole fluents """
    def __init__(self, fluent_names=None, obs_prefix=100, discount_factor=0.9, consistency_threshold=0.01):
        if fluent_names is None:
            fluent_names = [('x',), ('x_dot',), ('theta',), ('theta_dot',)]
        super().__init__(fluent_names, obs_prefix, discount_factor, consistency_threshold)

    def consistency_from_trace(self, simulation_trace: list, state_seq: list, delta_t: float = DEFAULT_DELTA_T):
        return self._consistency_from_matched_trace(simulation_trace, state_seq, delta_t)


class CartpolePlusPlusRepair(MetaModelRepair):
    """ Repairs Cartpole constants """
    def __init__(self, meta_model: MetaModel,
                 consistency_checker: DomainConsistency=CartpolePlusPlusConsistencyEstimator(),
                 consistency_threshold=settings.CP_CONSISTENCY_THRESHOLD):
        super().__init__(meta_model, consistency_checker)
        meta_model = CartPolePlusPlusMetaModel()
        self.fluents_to_repair = meta_model.repairable_constants
        self.repair_deltas = meta_model.repair_deltas
        self.meta_model_repair = GreedyBestFirstSearchContantFluentMetaModelRepair(meta_model, consistency_checker,self.fluents_to_repair,

                                                                                   self.repair_deltas,
                                                                                   consistency_threshold,
                                                                                   time_limit=settings.CP_REPAIR_TIMEOUT)

    """ Repair the given domain and plan such that the given plan's expected outcome matches the observed outcome"""
    def repair(self, observation, delta_t=1.0):
        return self.meta_model_repair.repair(observation, delta_t)
