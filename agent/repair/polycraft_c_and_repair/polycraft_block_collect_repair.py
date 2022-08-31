import logging

from agent.planning.polycraft_meta_model import PolycraftMetaModel
from agent.repair.meta_model_repair import AspectRepair
from agent.repair.polycraft_c_and_repair.polycraft_block_collection_consistency import \
    PolycraftBlockCollectOutcomeConsistency

from worlds.polycraft_world import BlockType

logger = logging.getLogger("Polycraft")


class PolycraftBlockCollectRepair(AspectRepair):
    """
    Repairs block-breaking outcomes in polycraft.
    At the moment, only repairs the outcome number. We could imagine a future implementation repairing _what_ is
    produced as well.
    """

    def __init__(self, consistency_estimator: PolycraftBlockCollectOutcomeConsistency):
        super().__init__(consistency_estimator.meta_model, consistency_estimator)

    def repair(self, observation, delta_t=1.0):
        """ Estimate consistency based on relevant values """
        self.meta_model.break_block_to_outcome[self.consistency_estimator.block_type] \
            = (self.meta_model.break_block_to_outcome[self.consistency_estimator.block_type][0],
               self.meta_model.break_block_to_outcome[self.consistency_estimator.block_type][1] +
               self.consistency_estimator.block_mismatch)
        description = f'block outcome {self.consistency_estimator.block_type} adjusted to: ' \
                      f'{self.meta_model.break_block_to_outcome[self.consistency_estimator.block_type]}'
        logger.info(description)
        return description, 0

    def __str__(self):
        return f'Polycraft block repair for {self.consistency_estimator.resource_type}'

    def __repr__(self):
        return f'Polycraft block repair for {self.consistency_estimator.resource_type}'
