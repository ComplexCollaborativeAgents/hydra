import logging

import settings
from agent.consistency.consistency_estimator import DomainConsistency
from agent.planning.meta_model import MetaModel
from agent.planning.sb_meta_model import ScienceBirdsMetaModel
from agent.repair.meta_model_repair import RepairModule, GreedyBestFirstSearchConstantFluentMetaModelRepair
from agent.repair.sb_consistency_estimators.sb_domain_consistency_estimator import ScienceBirdsConsistencyEstimator

logging.basicConfig(format='%(name)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("sb_repair")


class ScienceBirdsMetaModelRepair(RepairModule):
    """ The meta model repair used for ScienceBirds. """

    def __init__(self, meta_model: MetaModel = ScienceBirdsMetaModel(),
                 consistency_estimator: DomainConsistency = ScienceBirdsConsistencyEstimator(),
                 consistency_threshold=settings.SB_CONSISTENCY_THRESHOLD,
                 max_iterations=settings.SB_REPAIR_MAX_ITERATIONS, time_limit=settings.SB_REPAIR_TIMEOUT):
        super().__init__(meta_model, consistency_estimator)
        self.aspect_repair = [GreedyBestFirstSearchConstantFluentMetaModelRepair(
            meta_model, consistency_estimator, meta_model.repairable_constants, meta_model.repair_deltas,
            consistency_threshold=consistency_threshold, max_iterations=max_iterations, time_limit=time_limit)]

    def repair(self, observation, delta_t=1.0):
        # Right now, only one aspect is implemented
        return self.aspect_repair[0].repair(observation, delta_t)
