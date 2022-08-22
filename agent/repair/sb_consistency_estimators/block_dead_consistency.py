import settings
from agent.consistency.consistency_estimator import AspectConsistency


class BlockNotDeadConsistencyEstimator(AspectConsistency):
    """
    Checks consistency by considering which blocks are alive.
    Note that blocks being unexpectedly destroyed is not a strong indication of novelty, because we do not model block
    movement and secondary damage (only direct damage from the bird).
    """
    MIN_BLOCK_NOT_DEAD = 50
    MAX_INCREMENT = 50

    def __init__(self, fluent_names=None):
        if fluent_names is None:
            fluent_names = []
        super().__init__(fluent_names, fluent_template='block')

    def consistency_from_trace(self, simulation_trace: list, state_seq: list, delta_t: float = settings.SB_DELTA_T):
        """
        Estimate consistency by considering which blocks are or aren't destroyed.
        """
        last_state_in_sim = simulation_trace[-1][0]
        blocks_in_sim = last_state_in_sim.get_blocks()
        blocks_not_dead = self._objects_in_last_frame(simulation_trace, state_seq, in_sim=False, in_obs=True)
        return BlockNotDeadConsistencyEstimator.MIN_BLOCK_NOT_DEAD \
               + blocks_not_dead / len(blocks_in_sim) * BlockNotDeadConsistencyEstimator.MAX_INCREMENT
