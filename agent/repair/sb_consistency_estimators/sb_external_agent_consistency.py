from agent.consistency.consistency_estimator import AspectConsistency, DEFAULT_DELTA_T


class ExternalAgentLocationConsistencyEstimator(AspectConsistency):
    """
    Checks consistency by considering the location of the external agents in the problem. .
    """

    def __init__(self):
        super().__init__([], 'agent')

    def consistency_from_trace(self, simulation_trace: list, state_seq: list, delta_t: float = DEFAULT_DELTA_T):
        """ Estimate consistency by considering the location of the external agents in the observed state seq """
        return self._trajectory_compare(simulation_trace, state_seq, delta_t)
