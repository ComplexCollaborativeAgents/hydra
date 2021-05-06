from agent.repair.meta_model_repair import *
from agent.consistency.consistency_estimator import *
from agent.repair.focused_repair import *

logging.basicConfig(format='%(name)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("sb_repair")



''' Checks consistency by considering the location of the birds '''
class BirdLocationConsistencyEstimator(MetaModelBasedConsistencyEstimator):
    def __init__(self, unique_prefix_size = 100,discount_factor=0.9, consistency_threshold = 20):
        self.unique_prefix_size=unique_prefix_size
        self.discount_factor = discount_factor
        self.consistency_threshold = consistency_threshold

    ''' Estimate consitency by considering the location of the birds in the observed state seq '''
    def estimate_consistency(self, simulation_trace: list, state_seq: list, delta_t: float = DEFAULT_DELTA_T):
        # Get birds
        birds = set()
        for [state, _,_] in simulation_trace:
            if len(birds)==0: # TODO: Discuss how to make this work. Currently a new bird suddenly appears
                birds = state.get_birds()
            else:
                break

        fluent_names = []
        for bird in birds:
            fluent_names.append(('x_bird', bird))
            fluent_names.append(('y_bird', bird))

        consistency_checker = NumericFluentsConsistencyEstimator(fluent_names, self.unique_prefix_size,
                                                                 self.discount_factor,
                                                                 consistency_threshold = self.consistency_threshold)
        return consistency_checker.estimate_consistency(simulation_trace, state_seq, delta_t)

''' Checks consistency by considering which blocks are alive '''
class BlockNotDeadConsistencyEstimator(MetaModelBasedConsistencyEstimator):
    BLOCK_NOT_DEAD = 100

    ''' Estimate consitency by considering the location of the birds in the observed state seq '''
    def estimate_consistency(self, simulation_trace: list, state_seq: list, delta_t: float = DEFAULT_DELTA_T):
        last_state_in_obs = state_seq[-1]
        live_blocks_in_obs = last_state_in_obs.get_blocks()
        last_state_in_sim = simulation_trace[-1][0]
        blocks_in_sim = last_state_in_sim.get_blocks()
        # Make sure very block that is assumed to be dead in sim is indeed not alive in obs
        for block in blocks_in_sim:
            life_fluent = ('block_life', block)
            life_value = last_state_in_sim[life_fluent]
            if life_value <= 0:
                if block in live_blocks_in_obs:
                    logger.info("Block %s is alive but sim. thinks it is dead (life value=%.2f)" % (block, life_value))
                    return BlockNotDeadConsistencyEstimator.BLOCK_NOT_DEAD
        # TODO: Consider checking also the reverse condition

        return 0

''' Checks consistency for SB '''
class ScienceBirdsConsistencyEstimator(MetaModelBasedConsistencyEstimator):
    def __init__(self):
        self.consistency_estimators = list()
        self.consistency_estimators.append(BirdLocationConsistencyEstimator())
        self.consistency_estimators.append(BlockNotDeadConsistencyEstimator())

    def estimate_consistency(self, simulation_trace: list, state_seq: list, delta_t: float = DEFAULT_DELTA_T):
        max_inconsistency = 0
        for consistency_estimator in self.consistency_estimators:
            inconsistency = consistency_estimator.estimate_consistency(simulation_trace, state_seq, delta_t)
            if inconsistency > max_inconsistency:
                max_inconsistency = inconsistency

        return max_inconsistency

'''
 The meta model repair used for ScienceBirds. 
'''
class ScienceBirdsMetaModelRepair(GreedyBestFirstSearchMetaModelRepair):
    def __init__(self, meta_model = MetaModel(),
                 consistency_threshold=settings.SB_CONSISTENCY_THRESHOLD,
                 time_limit=settings.SB_REPAIR_TIMEOUT,
                 max_iterations=settings.SB_REPAIR_MAX_ITERATIONS):
        constants_to_repair = meta_model.repairable_constants
        repair_deltas = meta_model.repair_deltas
        consistency_estimator = ScienceBirdsConsistencyEstimator()
        super().__init__(constants_to_repair, consistency_estimator, repair_deltas,
                         consistency_threshold=consistency_threshold,
                         max_iterations=max_iterations,
                         time_limit=time_limit)

'''
 The meta model repair used for ScienceBirds. 
'''
class ScienceBirdsFocusedMetaModelRepair(FocusedMetaModelRepair):
    def __init__(self, meta_model = MetaModel(),
                 consistency_threshold=settings.SB_CONSISTENCY_THRESHOLD,
                 time_limit=settings.SB_REPAIR_TIMEOUT,
                 max_iterations=settings.SB_REPAIR_MAX_ITERATIONS):
        constants_to_repair = meta_model.repairable_constants
        repair_deltas = meta_model.repair_deltas
        consistency_estimator = ScienceBirdsConsistencyEstimator()
        super().__init__(constants_to_repair, consistency_estimator, repair_deltas,
                         consistency_threshold=consistency_threshold,
                         max_iterations=max_iterations,
                         time_limit=time_limit)