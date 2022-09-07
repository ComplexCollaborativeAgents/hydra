import logging
from typing import List, Tuple

import settings
from agent.consistency.consistency_estimator import AspectConsistency, DEFAULT_DELTA_T
from agent.planning.nyx.compiler.preconditions_tree import Happening
from agent.planning.pddl_plus import PddlPlusState
from agent.planning.polycraft_meta_model import PolycraftMetaModel
from agent.planning.polycraft_planning.polycraft_pddl_objects_and_constants import PddlGameMapCellType, Function
from agent.planning.polycraft_planning.actions import BreakAndCollect, TeleportToBreakAndCollect
from utils.polycraft_utils import coordinates_to_cell
from worlds.polycraft_world import PolycraftAction, BlockType

logger = logging.getLogger("Polycraft")


class PolycraftBlockCollectOutcomeConsistency(AspectConsistency):
    """
    Finds inconsistency for breaking blocks giving an unexpected number of pieces.
    TODO: should be per block type? or one for all types?
    """
    def __init__(self, resource_type, metamodel: PolycraftMetaModel):
        super().__init__([])
        self.resource_type = resource_type
        self.block_type = ''
        for b_type in BlockType:
            if b_type.name.lower().find(self.resource_type) > -1:
                self.block_type = b_type.value
                break
        self.meta_model = metamodel
        self.block_mismatch = 0

    def consistency_from_trace(self, simulation_trace: List, observation_trace: List[PddlPlusState], pddl_plan=None,
                               delta_t: float = settings.POLYCRAFT_DELTA_T):
        """ Estimate consistency based on relevant values """
        self.block_mismatch = 0
        ready_to_compare = False
        initial_mismatch = 0

        logger.info(
            f"sim still, steve:{simulation_trace[0][0].numeric_fluents[('Steve_x',)]},{simulation_trace[0][0].numeric_fluents[('Steve_z',)]} ")

        # Get relevant fluent names
        if not self.fluent_names:
            for fl in simulation_trace[0][0].numeric_fluents.keys():
                if fl[0].startswith('count_') and fl[0].endswith(self.resource_type):
                    self.fluent_names.append(fl)
                    break

        # Get relevant action names
        simulation_trace.insert(0, observation_trace[0])

        for sim_tuple, obs_state, action in zip(simulation_trace, observation_trace, pddl_plan):
            sim_state = sim_tuple[0]
            logger.info(f"action {sim_tuple[2]}, planned action {action}, time {sim_tuple[1]}")

            # Is this an action related to this aspect
            is_collection_action = action.poly_action.__class__.__name__.lower().find('collect') > -1
            # Yoni: The above line is horrible. Is there a better way to do it?
            logger.info(action.action_name.lower())

            if is_collection_action:
                # fluent_name = (Function.cell_type.name, PddlGameMapCellType.get_cell_object_name(action.poly_action.cell))
                state_diff = sim_state.diff(obs_state)
                for diff_key, diff_value in state_diff.items():
                    logger.info(f"\t diff key {diff_key}, diff_value {diff_value}")
                # logger.info(f"Action's cell fluent name {fluent_name}")
                # logger.info(f"Sim_state cell type {sim_state.numeric_fluents.get(fluent_name)}")
                # logger.info(f"Has cell?{fluent_name in sim_state}")
                # if fluent_name in sim_state:
                #     for a_fluent_name, a_fluent_value in sim_state.numeric_fluents.items():
                #         logger.info(f"\t {a_fluent_name} {a_fluent_value}")
                # logger.info(f"Obs_state cell type {obs_state.numeric_fluents.get(fluent_name)}")
                # logger.info(f"Has cell?{fluent_name in obs_state}")
                # logger.info(f"Expected type {self.meta_model.block_type_to_idx[self.block_type]}")

            if is_collection_action and \
                    sim_state.numeric_fluents[(Function.cell_type.name, PddlGameMapCellType.get_cell_object_name(action.poly_action.cell))] == \
                    self.meta_model.block_type_to_idx[self.block_type]:
                initial_mismatch = sim_state.numeric_fluents[self.fluent_names[0]] - \
                                   obs_state.numeric_fluents[self.fluent_names[0]]
                logger.info(f'initial: Simulation: {sim_state.numeric_fluents[self.fluent_names[0]]}, observation: {obs_state.numeric_fluents[self.fluent_names[0]]}')
                ready_to_compare = True
            elif ready_to_compare:
                end_mismatch = sim_state.numeric_fluents[self.fluent_names[0]] - \
                               obs_state.numeric_fluents[self.fluent_names[0]]
                logger.info(f'second: Simulation: {sim_state.numeric_fluents[self.fluent_names[0]]}, observation: {obs_state.numeric_fluents[self.fluent_names[0]]}')

                self.block_mismatch += end_mismatch - initial_mismatch
                initial_mismatch = 0
                ready_to_compare = False
                # break
                # Return the result of a single mismatch. We don't really get more information by summing over multiple
                #  breakings of the same type of brick.

        return abs(self.block_mismatch)

    def __str__(self):
        return f'Polycraft block consistency for {self.resource_type}'

    def __repr__(self):
        return f'Polycraft block consistency for {self.resource_type}'

