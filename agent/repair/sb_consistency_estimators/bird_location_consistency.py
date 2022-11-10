from agent.consistency.consistency_estimator import AspectConsistency, DEFAULT_DELTA_T


class BirdLocationConsistencyEstimator(AspectConsistency):
    """
    Checks consistency by considering the location of the birds.
    """

    def __init__(self):
        super().__init__([], 'bird')

    def consistency_from_trace(self, simulation_trace: list, state_seq: list, pddl_plan=None,
                               delta_t: float = DEFAULT_DELTA_T):
        """
        Estimate consistency by considering the location of the birds in the observed state seq.
        """
        if pddl_plan is None:
            pddl_plan = []
        return self._trajectory_compare(simulation_trace, state_seq, delta_t)
