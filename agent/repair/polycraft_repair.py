import logging

import settings
from agent.consistency.nyx_pddl_simulator import NyxPddlPlusSimulator
from agent.repair.meta_model_repair import GreedyBestFirstSearchConstantFluentMetaModelRepair, RepairModule
from agent.repair.polycraft_c_and_repair.polycraft_block_collect_repair import PolycraftBlockCollectRepair
from agent.repair.polycraft_c_and_repair.polycraft_domain_consistency_estimator import \
    PolycraftConsistencyEstimator
from numpy import argmax

logger = logging.getLogger("Polycraft")


class PolycraftMetaModelRepair(RepairModule):
    """ The meta model repair used for Polycraft. """

    def __init__(self, meta_model, consistency_threshold=settings.POLYCRAFT_CONSISTENCY_THRESHOLD,
                 time_limit=settings.POLYCRAFT_REPAIR_TIMEOUT, max_iterations=settings.POLYCRAFT_REPAIR_MAX_ITERATIONS):
        super().__init__(meta_model, PolycraftConsistencyEstimator(meta_model))
        self.consistency_threshold = consistency_threshold
        self.aspect_repair = [GreedyBestFirstSearchConstantFluentMetaModelRepair(self.meta_model,
                                                                                 self.consistency_estimator,
                                                                                 meta_model.repairable_constants,
                                                                                 meta_model.repair_deltas,
                                                                                 settings.POLYCRAFT_CONSISTENCY_THRESHOLD,
                                                                                 max_iterations=100, time_limit=600)]
        # self.aspect_repair = [PolycraftBlockCollectRepair(c_e) for c_e in
        #                       self.consistency_estimator.block_outcome_estimators]

    def repair(self, observation, delta_t=1.0):
        descriptions = ''
        max_ic_index = argmax(self.consistency_estimator.latest_inconsistencies)
        repair_attemtps = 0
        max_repair_attempts = len(self.aspect_repair)  # TODO seems reasonable for now
        while max(self.consistency_estimator.latest_inconsistencies) > self.consistency_threshold and \
                repair_attemtps < max_repair_attempts:
            description, _ = self.aspect_repair[max_ic_index].repair(observation, delta_t)
            descriptions += str(description)
            self.consistency_estimator.consistency_from_observations(self.meta_model, NyxPddlPlusSimulator(),
                                                                     observation, delta_t)
            max_ic_index = argmax(self.consistency_estimator.latest_inconsistencies)
            repair_attemtps += 1

        return descriptions, max(self.consistency_estimator.latest_inconsistencies)
