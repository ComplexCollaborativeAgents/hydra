import random

import settings
from agent.hydra_agent import HydraPlanner
from agent.planning.polycraft_meta_model import PolycraftMetaModel
from agent.planning.polycraft_planning.actions import CreateByRecipe, CollectAndMineItem, CreateByTrade, MacroAction, \
    TeleportAndFaceCell
from agent.polycraft_hydra_agent import logger
from utils.polycraft_utils import get_adjacent_cells, is_adjacent_to_steve, get_angle_to_adjacent_cell
from worlds.polycraft_world import PolycraftState, ItemType, PolyNoAction, BlockType, PolycraftAction, PolyPlaceTreeTap, \
    PolyCollect


class FixedPlanPlanner(HydraPlanner):
    ''' Planner for the polycraft domain that follows the following fixed plan, specified in the CreateWoodenPogoStick macro action '''
    def __init__(self, meta_model = PolycraftMetaModel(), planning_path = settings.POLYCRAFT_PLANNING_DOCKER_PATH):
        super().__init__(meta_model)

    def make_plan(self,state: PolycraftState):
        # Stopping condition
        if state.count_items_of_type(ItemType.WOODEN_POGO_STICK.value)>0:
            logger.info("Already have the pogo stick! why plan?")
            return [PolyNoAction()]

        return [CreateWoodenPogoStick()]


class CreateWoodenPogoStick(CreateByRecipe):
    ''' Do whatever is needed to create a wooden pogo stick
    Designed mostly for the following recipe:
        1) Mine diamonds and craft 2 diamons blocks
        2) Mine logs and make 2 sticks.
        3) Mine platinum and trade to titanium blocks
        4) Obtain the pallets somehow
    '''
    def __init__(self):
        super().__init__(item_to_craft = ItemType.WOODEN_POGO_STICK.value,
            needs_crafting_table=True,
            needs_iron_pickaxe=False)

    def _get_action_to_collect_ingredient(self, ingredient:str, quantity:int):
        ''' An action that is used to collect the given quantity of the given ingredient '''
        if ingredient == ItemType.DIAMOND_BLOCK.value:
            return CreateDiamondBlock()
        elif ingredient == ItemType.STICK.value:
            return CreateStick()
        elif ingredient == ItemType.BLOCK_OF_TITANIUM.value:
            return CreateBlockOfTitanium()
        elif ingredient == ItemType.PLANKS.value:
            return CreatePlanks()
        elif ingredient == ItemType.SACK_POLYISOPRENE_PELLETS.value:
            return CreateSackPolyisoprenePellets()
        else:
            raise ValueError(f"Unknown ingredient for {self.item_to_craft}: {ingredient}")


class CreateDiamondBlock(CreateByRecipe):
    ''' Do whatever is needed to craft a diamond block.
    This may include selecting an iron pickaxe and mining diamond ore '''
    def __init__(self):
        super().__init__(item_to_craft = ItemType.DIAMOND_BLOCK.value,
                         needs_crafting_table=True,
                         needs_iron_pickaxe=True)

    def _get_action_to_collect_ingredient(self, ingredient:str, quantity:int):
        ''' An action that is used to collect the given quantity of the given ingredient '''
        assert(ingredient==ItemType.DIAMOND.value)
        return CollectAndMineItem(ingredient, quantity, [BlockType.DIAMOND_ORE.value])


class CreateStick(CreateByRecipe):
    ''' Do whatever is needed to craft a stick '''
    def __init__(self):
        super().__init__(item_to_craft=ItemType.STICK.value,
                         needs_crafting_table=False,
                         needs_iron_pickaxe=False)

    def _get_action_to_collect_ingredient(self, ingredient:str, quantity:int):
        ''' An action that is used to collect the given quantity of the given ingredient '''
        assert(ingredient==ItemType.PLANKS.value)
        return CreatePlanks()


class CreateBlockOfTitanium(CreateByTrade):
    ''' Create a block of titanium by following these steps:
        1. Collect platinum
        2. Tradefor titanium
    '''
    def __init__(self):
        super().__init__(item_type_to_get=ItemType.BLOCK_OF_TITANIUM.value, item_types_to_give = [BlockType.BLOCK_OF_PLATINUM.value])

    def _get_action_to_collect_ingredient(self, ingredient:str, quantity:int):
        ''' An action that is used to collect the given quantity of the given ingredient '''
        if ingredient == BlockType.BLOCK_OF_PLATINUM.value:
            return CollectAndMineItem(desired_item_type=BlockType.BLOCK_OF_PLATINUM.value,
                                      desired_quantity=quantity,
                                      relevant_block_types=[BlockType.BLOCK_OF_PLATINUM.value],
                                      needs_iron_pickaxe=True)
        else:
            raise ValueError(f"Unknown ingredient for {self.item_type_to_get}: {ingredient}")


class CreatePlanks(CreateByRecipe):
    ''' Do whatever is needed to craft a stick '''
    def __init__(self):
        super().__init__(item_to_craft = ItemType.PLANKS.value,
            needs_crafting_table=False,
            needs_iron_pickaxe=False)

    def _get_action_to_collect_ingredient(self, ingredient:str, quantity:int):
        ''' An action that is used to collect the given quantity of the given ingredient '''
        assert(ingredient==ItemType.LOG.value)
        return CollectAndMineItem(ingredient, quantity, [BlockType.LOG.value])


class CreateSackPolyisoprenePellets(MacroAction):
    ''' Create a sack of polyisoprene pellets by following these steps:
        1. Craft tree tap
        2. Place tree tap
        3. Collect sack of polyisoprene pellets from tree tap
    '''
    ''' An action that crafts an item by following a recipe. If ingredients are missing it goes to search and collect them '''
    def __init__(self):
        super().__init__()

    def _get_next_action(self)->PolycraftAction:
        ''' Return an action to perform '''
        state = self._current_state

        tree_tap_cells = state.get_cells_of_type(ItemType.TREE_TAP.value)
        if len(tree_tap_cells)==0: # Need to place the tree tap
            if state.count_items_of_type(ItemType.TREE_TAP.value)==0: # Need to create the tree tap
                return CreateTreeTap()
            else:
                # Find place for a tree tap
                log_cells = state.get_cells_of_type(BlockType.LOG.value, only_accessible=True)
                assert (len(log_cells) > 0)
                cells_to_place_tree_tap = []
                for cell in log_cells:
                    for adjacent_cell in get_adjacent_cells(cell):
                        if adjacent_cell in state.game_map and \
                                state.game_map[adjacent_cell]["name"] == BlockType.AIR.value and \
                                state.game_map[adjacent_cell]["isAccessible"]:
                            cells_to_place_tree_tap.append(adjacent_cell)
                assert (len(cells_to_place_tree_tap) > 0)

                # Check if there's a relevant cell we're looking at
                cell_to_place = None
                for cell in cells_to_place_tree_tap:
                    if is_adjacent_to_steve(cell, state) and get_angle_to_adjacent_cell(cell, state)==0:
                        cell_to_place = cell
                        break
                # If Steve isn't facing a relevant cell, teleport to one
                if cell_to_place is None:
                    cell = cells_to_place_tree_tap[0]
                    return TeleportAndFaceCell(cell)
                else: # Steve is facing a relevant cell! just place the tree tap
                    return PolyPlaceTreeTap()

        # Tree tap exists, go and collect!
        if state.is_facing_type(ItemType.TREE_TAP.value):
            self._is_done = True
            return PolyCollect()
        else: # Move to a tree tap cell and collect
            cell = random.choice(tree_tap_cells)
            return TeleportAndFaceCell(cell)


class CreateTreeTap(CreateByRecipe):
    ''' Do whatever is needed to create a tree tap '''
    def __init__(self):
        super().__init__(item_to_craft = ItemType.TREE_TAP.value,
            needs_crafting_table=True,
            needs_iron_pickaxe=False)

    def _get_action_to_collect_ingredient(self, ingredient:str, quantity:int):
        ''' An action that is used to collect the given quantity of the given ingredient '''
        if ingredient == ItemType.STICK.value:
            return CreateStick()
        elif ingredient == ItemType.PLANKS.value:
            return CreatePlanks()
        else:
            raise ValueError(f"Unknown ingredient for {self.item_to_craft}: {ingredient}")