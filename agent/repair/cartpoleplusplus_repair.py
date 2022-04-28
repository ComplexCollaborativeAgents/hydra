from agent.consistency.sequence_consistency_estimator import SequenceConsistencyEstimator
from agent.planning.cartpoleplusplus_pddl_meta_model import *
from agent.repair.meta_model_repair import *

''' Checks consistency by considering the location of the Cartpole fluents '''
class CartpolePlusPlusConsistencyEstimator(MetaModelBasedConsistencyEstimator):
    def __init__(self, unique_prefix_size = 100,discount_factor=0.9, consistency_threshold = 0.01):
        self.unique_prefix_size=unique_prefix_size
        self.discount_factor = discount_factor
        self.consistency_threshold = consistency_threshold

    ''' Estimate consitency by considering the location of the birds in the observed state seq '''
    def estimate_consistency(self, simulation_trace: list, state_seq: list, delta_t: float = DEFAULT_DELTA_T):
        fluent_names = []
        fluent_names.append(('pos_x',))
        fluent_names.append(('pos_y',))
        fluent_names.append(('theta_x',))
        fluent_names.append(('theta_y',))

        # Get blocks
        blocks = set()
        for [state, _, _] in simulation_trace:
            if len(blocks) == 0:  # TODO: Discuss how to make this work. Currently a new bird suddenly appears
                blocks = state.get_cp_blocks()
            else:
                break

        for bl in blocks:
            fluent_names.append(('block_x', bl))
            fluent_names.append(('block_y', bl))
            fluent_names.append(('block_z', bl))

        consistency_checker = SequenceConsistencyEstimator(fluent_names, self.unique_prefix_size, self.discount_factor, self.consistency_threshold)
        return consistency_checker.estimate_consistency(simulation_trace, state_seq, delta_t)


class CartpolePlusPlusRepair(GreedyBestFirstSearchMetaModelRepair):
    ''' Repairs Cartpole++ constants '''
    def __init__(self, consistency_checker=CartpolePlusPlusConsistencyEstimator(),
                 consistency_threshold=settings.CP_CONSISTENCY_THRESHOLD):
        super().__init__(consistency_checker, consistency_threshold, time_limit=settings.CP_REPAIR_TIMEOUT)


    def get_repair_as_json(self, repair):
        ''' Get a JSON representation of the given repair '''
        return json.dumps(dict(zip(self.meta_model_repair.fluents_to_repair, repair)))