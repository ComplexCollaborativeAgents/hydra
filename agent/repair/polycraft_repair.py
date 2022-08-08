import settings
from agent.consistency.consistency_estimator import DEFAULT_DELTA_T, AspectConsistency, DomainConsistency
from agent.consistency.observation import HydraObservation
from agent.repair.meta_model_repair import GreedyBestFirstSearchContantFluentMetaModelRepair


class PolycraftConsistencyEstimator(DomainConsistency):
    """
    Total inconsistency for a polycraft game.
    TODO: nearby blocks? suggeted repair fluents??
    """
    def __init__(self):
        super().__init__([PolycraftInventoryConsistency()])


class PolycraftInventoryConsistency(AspectConsistency):
    def __init__(self, fluent_names=None, obs_prefix=100, discount_factor=0.9, consistency_threshold=0.01):
        if fluent_names is None:
            fluent_names = [('selectedItem',)]
        super().__init__(fluent_names, obs_prefix, discount_factor, consistency_threshold)

    def consistency_from_trace(self, simulation_trace: list, state_seq: list, delta_t: float = DEFAULT_DELTA_T):
        """ Estimate consistency based on relevant values """
        # Need to recalculate items to track, because they might change from level to level and between tasks.
        # TODO figure out fluents to track properly
        self.fluent_names = []
        self.fluent_names.append(('selectedItem',))
        self.fluent_names.extend(
            (fl[0],) for fl in simulation_trace[0].state.numeric_fluents.keys() if fl[0].startswith('count_'))
        # How is our agent's position stored?
        return self.consistency_from_matched_trace(simulation_trace, state_seq, delta_t)


class PolycraftMetaModelRepair(GreedyBestFirstSearchContantFluentMetaModelRepair):
    """ The meta model repair used for ScienceBirds. """

    def __init__(self, meta_model,
                 consistency_threshold=settings.POLYCRAFT_CONSISTENCY_THRESHOLD,
                 time_limit=settings.POLYCRAFT_REPAIR_TIMEOUT,
                 max_iterations=settings.POLYCRAFT_REPAIR_MAX_ITERATIONS):
        constants_to_repair = meta_model.repairable_constants
        repair_deltas = meta_model.repair_deltas
        consistency_estimator = PolycraftConsistencyEstimator()
        super().__init__(meta_model, consistency_estimator,constants_to_repair,  repair_deltas,
                         consistency_threshold=consistency_threshold,
                         max_iterations=max_iterations,
                         time_limit=time_limit)

