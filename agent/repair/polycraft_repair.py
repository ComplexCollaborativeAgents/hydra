import settings
from agent.consistency.consistency_estimator import ConsistencyEstimator, DEFAULT_DELTA_T
from agent.consistency.observation import HydraObservation
from agent.consistency.pddl_plus_simulator import PddlPlusSimulator, InconsistentPlanError
from agent.consistency.sequence_consistency_estimator import SequenceConsistencyEstimator
from agent.planning.polycraft_meta_model import PolycraftMetaModel
from agent.repair.meta_model_repair import GreedyBestFirstSearchMetaModelRepair


class MetaModelBasedConsistencyEstimator(ConsistencyEstimator):
    PLAN_FAILED_CONSISTENCY_VALUE = 1000  # A constant representing the inconsistency value of a meta model in which the executed plan is inconsistent

    def compute_consistency(self, observation, meta_model, simulator: PddlPlusSimulator, delta_t):
        """ Computes the consistency of a given observation w.r.t the given meta model using the given simulator """

        try:
            expected_trace, plan = simulator.get_expected_trace(observation, meta_model, delta_t)
            observed_seq = observation.get_pddl_states_in_trace(meta_model)
            consistency = self.estimate_consistency(expected_trace, observed_seq, delta_t)
        except InconsistentPlanError:  # Sometimes the repair makes the executed plan be inconsistent, e.g., its preconditions are not satisfied
            consistency = MetaModelBasedConsistencyEstimator.PLAN_FAILED_CONSISTENCY_VALUE
        return consistency


class PolycraftConsistencyEstimator(MetaModelBasedConsistencyEstimator):
    def __init__(self, unique_prefix_size=100, discount_factor=0.9, consistency_threshold=0.01):
        self.unique_prefix_size = unique_prefix_size
        self.discount_factor = discount_factor
        self.consistency_threshold = consistency_threshold

    def estimate_consistency(self, simulation_trace: list, state_seq: list, delta_t: float = DEFAULT_DELTA_T):
        """ Estimate consistency based on relevant values """
        fluent_names = []
        fluent_names.extend(
            (fl[0],) for fl in simulation_trace[0].state.numeric_fluents.keys() if fl[0].startswith('count_'))
        fluent_names.append(('selectedItem',))
        # How is our agent's position stored?
        consistency_checker = SequenceConsistencyEstimator(fluent_names, self.unique_prefix_size, self.discount_factor,
                                                           self.consistency_threshold)
        return consistency_checker.estimate_consistency(simulation_trace, state_seq, delta_t)


class PolycraftMetaModelRepair(GreedyBestFirstSearchMetaModelRepair):
    """ The meta model repair used for ScienceBirds. """

    def __init__(self, meta_model=PolycraftMetaModel(),
                 consistency_threshold=settings.POLYCRAFT_CONSISTENCY_THRESHOLD,
                 time_limit=settings.POLYCRAFT_REPAIR_TIMEOUT,
                 max_iterations=settings.POLYCRAFT_REPAIR_MAX_ITERATIONS):
        constants_to_repair = meta_model.repairable_constants
        repair_deltas = meta_model.repair_deltas
        consistency_estimator = PolycraftConsistencyEstimator()
        super().__init__(constants_to_repair, consistency_estimator, repair_deltas,
                         consistency_threshold=consistency_threshold,
                         max_iterations=max_iterations,
                         time_limit=time_limit)

    def compute_consistency(self, repair: list, observation: HydraObservation, max_iterations=50):
        return super().compute_consistency(repair, observation, max_iterations)
