import settings
from agent.consistency.consistency_estimator import AspectConsistency, DEFAULT_DELTA_T


class PigDeadConsistencyEstimator(AspectConsistency):
    """
    Are the pigs we expected to be dead actually dead?
    Note that pigs being unexpectedly killed is not a strong indication of novelty, because we do not model block
    movement and secondary damage (only direct damage from the bird).
    """

    def __init__(self, fluent_names=None):
        if fluent_names is None:
            fluent_names = []
        super().__init__(fluent_names, fluent_template='pig')

    def consistency_from_trace(self, simulation_trace: list, state_seq: list, pddl_plan=None,
                               delta_t: float = DEFAULT_DELTA_T):
        """
        simulation_trace: list of (state, time, actions) lists in simulation (expected plan)
        state_seq: observations (what actually happened)
        delta_t: time step for simulation
        """
        if pddl_plan is None:
            pddl_plan = []
        pigs_not_dead = self._objects_in_last_frame(simulation_trace, state_seq, in_sim=True, in_obs=False)
        return pigs_not_dead * 50
