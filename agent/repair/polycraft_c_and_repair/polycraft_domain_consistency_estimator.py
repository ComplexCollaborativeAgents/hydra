from agent.consistency.consistency_estimator import DomainConsistency
from agent.repair.polycraft_c_and_repair.polycraft_block_collection_consistency import \
    PolycraftBlockCollectOutcomeConsistency
from agent.repair.polycraft_c_and_repair.polycraft_inventory_consistency import PolycraftInventoryConsistency

from agent.planning.polycraft_meta_model import PolycraftMetaModel


class PolycraftConsistencyEstimator(DomainConsistency):
    """
    Total inconsistency for a polycraft game.
    TODO: nearby blocks? suggested repair fluents??
    """

    def __init__(self, meta_model: PolycraftMetaModel):
        self.aspect_estimators = [PolycraftInventoryConsistency()]
            # PolycraftBlockCollectOutcomeConsistency('log', meta_model),
            #                              PolycraftBlockCollectOutcomeConsistency('diamond', meta_model),
            #                              ]

        super().__init__(self.aspect_estimators)

    def consistency_from_observations(self, meta_model, simulator, observation, delta_t):
        return super().consistency_from_observations(meta_model, simulator, observation, delta_t)

