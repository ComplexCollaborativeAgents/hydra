import logging

from agent.consistency.consistency_estimator import AspectConsistency, DEFAULT_DELTA_T

logger = logging.getLogger("Polycraft")


class PolycraftInventoryConsistency(AspectConsistency):
    def __init__(self, fluent_names=None, obs_prefix=100, discount_factor=0.9, consistency_threshold=0.01):
        if fluent_names is None:
            fluent_names = []
        super().__init__(fluent_names, obs_prefix=obs_prefix, discount_factor=discount_factor,
                         consistency_threshold=consistency_threshold)

    def consistency_from_trace(self, simulation_trace: list, state_seq: list, pddl_plan=None,
                               delta_t: float = DEFAULT_DELTA_T):
        """ Estimate consistency based on relevant values """
        # Need to recalculate items to track, because they might change from level to level and between tasks.
        # Note that the 'selected item' is not modeled correctly by the PDDL, and should not be tracked.
        # TODO figure out fluents to track properly
        if pddl_plan is None:
            pddl_plan = []
        self.fluent_names = []
        self.fluent_names.extend(
            (fl[0],) for fl in simulation_trace[0].state.numeric_fluents.keys() if fl[0].startswith('count_'))
        logger.info(self.fluent_names)
        # How is our agent's position stored?
        return self._consistency_from_matched_trace(simulation_trace, state_seq, delta_t)
