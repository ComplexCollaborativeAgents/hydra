from agent.repair.meta_model_repair import *


class CartpoleConsistencyEstimator(ConsistencyEstimator):
    """ Checks consistency by considering the location of the Cartpole fluents """
    def __init__(self, fluent_names=None, obs_prefix=100, discount_factor=0.9, consistency_threshold=0.01):
        if fluent_names is None:
            fluent_names = [('x',), ('x_dot',), ('theta',), ('theta_dot',)]
        super().__init__(fluent_names, obs_prefix, discount_factor, consistency_threshold)


class CartpoleRepair(MetaModelRepair):
    """ Repairs Cartpole constants """
    def __init__(self, consistency_checker=CartpoleConsistencyEstimator(),
                 consistency_threshold=settings.CP_CONSISTENCY_THRESHOLD):
        meta_model = CartPoleMetaModel()
        self.fluents_to_repair = meta_model.repairable_constants
        self.repair_deltas = meta_model.repair_deltas
        self.meta_model_repair = GreedyBestFirstSearchMetaModelRepair(self.fluents_to_repair,
                                                                      consistency_checker,
                                                                      self.repair_deltas,
                                                                      consistency_threshold)

    """ Repair the given domain and plan such that the given plan's expected outcome matches the observed outcome"""
    def repair(self, pddl_meta_model, observation, delta_t=1.0):
        return self.meta_model_repair.repair(pddl_meta_model, observation, delta_t)
