import random

from utils.polycraft_utils import *
from worlds.polycraft_world import *
import worlds.polycraft_interface.client.polycraft_interface as poly


class MacroAction(PolycraftAction):
    ''' A macro action is a generator of basic PolycraftActions based on the current state '''
    def __init__(self):
        super().__init__()
        self.active_action = None # If we're in the middle of doing some action
        self._is_done = False
        self._current_state = None

    def _get_next_action(self)->PolycraftAction:
        raise NotImplementedError("Subclasses of MacroAction should implement this and set self.next_action in it")

    def is_done(self):
        return self._is_done

    def set_current_state(self, state: PolycraftState):
        self._current_state = state
        if self.active_action is not None:
            self.active_action.set_current_state(state)

    def do(self, env: poly.PolycraftInterface) -> dict:
        ''' Key: macro action accept the environment, not the polycraft_interface'''
        logger.info(f"Doings macro action {self}")
        if self.active_action is None:
            next_action = self._get_next_action()
        else:
            next_action = self.active_action

        if isinstance(next_action, MacroAction):
            next_action.set_current_state(self._current_state)

        result = next_action.do(env)
        self.success = next_action.is_success(result)

        # If next action not done yet, do it
        if isinstance(next_action, MacroAction):
            if next_action.is_done()==False:
                self.active_action = next_action
            else:
                logger.info(f"Macro action {self.active_action} is done")
                self.active_action = None
        else:
            self.active_action = None
        return result

class TeleportAndFaceCell(PolycraftAction):
    ''' Macro for teleporting to a given cell and turning to face it '''
    def __init__(self, cell: str):
        super().__init__()
        self.cell = cell

    def __str__(self):
        return "<TeleportAndFaceCell {} success={}>".format(self.cell, self.success)

    def do(self, poly_client: poly.PolycraftInterface) -> dict:
        tp_action = PolyTP(self.cell, dist=1)
        result = tp_action.do(poly_client)
        if tp_action.success == False:
            logger.info(f"teleport_and_face_cell({self.cell}) failed during TP_TO_POS, Message: {result}")
            return result

        # Orient so we face the block
        current_state = PolycraftState.create_current_state(poly_client)
        turn_angle = get_angle_to_adjacent_cell(self.cell, current_state)
        if turn_angle == 0:
            self.success = self.is_success(result)
            return result
        else:
            turn_action = PolyTurn(turn_angle)
            result = turn_action.do(poly_client)
            self.success = turn_action.is_success(result)
            self.command_result = result
            if self.success == False:
                logger.info(f"teleport_and_face_cell({self.cell}) failed during TURN, Message: {result}")
            else:
                self.success = True
            return result

class PolyBreakAndCollect(PolycraftAction):
    """ Teleport near a brick, break it, and collect the resulting item """
    def __init__(self, cell: str):
        super().__init__()
        self.cell = cell

    def __str__(self):
        return "<PolyBreakAndCollect {} success={}>".format(self.cell, self.success)

    def do(self, poly_client: poly.PolycraftInterface) -> dict:
        # Move
        action = TeleportAndFaceCell(self.cell)
        result = action.do(poly_client)
        if action.is_success(result)==False:
            return result

        # Store the state before breaking, to identify the new item
        current_state = PolycraftState.create_current_state(poly_client)

        # Break the block!
        result = poly_client.BREAK_BLOCK()
        if self.is_success(result) == False:
            self.success = False
            logger.info(f"Action {str(self)} failed during BREAK_BLOCK, Message: {result}")
            return result

        # Find and collect the item
        previous_state = current_state
        current_state = PolycraftState.create_current_state(poly_client)
        # If item in inventory - success!
        self._wait_to_collect_adjacent_items(current_state, poly_client)
        current_state = PolycraftState.create_current_state(poly_client)
        state_diff = current_state.diff(previous_state)
        if has_new_item(state_diff):
            self.success = True
            return result

        # Else, new item appears as an EntityItem and needs to be collected
        assert("entities" in state_diff) # If the new item is not in the inventory, it should be a new EntityItem
        new_entity_items = get_new_entity_items(state_diff)
        assert(len(new_entity_items)>0)

        if len(new_entity_items)>=1: # Choose the closest entity item
            min_dist_to_item = None
            new_item_pos = None
            new_item = None
            steve_pos = current_state.location['pos']
            for (entity_id, entity_attr) in new_entity_items:
                item_pos = entity_attr['pos']
                if new_item_pos is None:
                    min_dist_to_item = compute_cell_distance(steve_pos, item_pos)
                    new_item_pos = item_pos
                    new_item = entity_id
                else:
                    dist_to_item = compute_cell_distance(steve_pos, item_pos)
                    if dist_to_item<min_dist_to_item:
                        min_dist_to_item = dist_to_item
                        new_item_pos = item_pos
                        new_item = entity_id

        # Move to new item location to collect it
        item_pos_cell = ",".join([str(coord) for coord in new_item_pos])
        logger.info(f"Item not in inventory, teleport to its cell: {item_pos_cell}")
        result = poly_client.TP_TO_ENTITY(new_item)
        if self.is_success(result) == False:
            self.success = False
            logger.info(f"Action {str(self)} failed during TP_TO_ENTITY(new_item), Message: {result}")
            return result

        # Assert new item collected
        previous_state = current_state
        current_state = PolycraftState.create_current_state(poly_client)
        self._wait_to_collect_adjacent_items(current_state, poly_client)
        current_state = PolycraftState.create_current_state(poly_client)
        state_diff = current_state.diff(previous_state)
        assert(has_new_item(state_diff))

        self.success = True
        return result

    def _wait_to_collect_adjacent_items(self,current_state:PolycraftState, poly_client: poly.PolycraftInterface):
        ''' Waits some time steps if there is item near by that should have been collected automatically '''
        MAX_WAIT = 3 # The maximal number of no-ops we allow before giving up
        REACHABILITY = 1 # If the item is at this distance from Steve, we expect it to be automatically collected
        steve_location = current_state.location["pos"]
        for i in range(MAX_WAIT):
            has_item_in_range = False
            items = current_state.get_entities_of_type(EntityType.ITEM.value)
            for entity_id in items:
                entity_attr = current_state.entities[entity_id]
                item_location = entity_attr["pos"]
                distance = compute_cell_distance(steve_location, item_location)
                if distance<=REACHABILITY:
                    has_item_in_range = True
                    break

            # if no item is in range, no point in waiting
            if has_item_in_range==False:
                return
            else:
                poly_client.CHECK_COST() # Do a no-op


class CollectAndMineItem(PolycraftAction):
    ''' A high-level macro action that accepts the desired number of items to collect and which blocks to mine to get it.
    Pseudo code:
    Input: desired_item, desired_count, relevant_block_types_to_mine
    While inventory does not contain the desired item in the desired amount
        If EntityItems already exists in reachable cells
            Teleport to these cells to collect them
        If there are accessible cells of the relevant block to mine
            Teleport to these cells and mine the desired item
        Otherwise, choose an accessible block and mine it
    '''
    def __init__(self, desired_item_type: str, desired_quantity:int, relevant_block_types:list, max_tries = 5, needs_iron_pickaxe=False):
        super().__init__()
        self.desired_item_type = desired_item_type
        self.desired_quantity = desired_quantity
        self.relevant_block_types = relevant_block_types
        self.max_tries = max_tries # Declare failure if after max_tries iterations of collecting and mining blocks the desired quantity hasn't been reached.
        self.needs_iron_pickaxe=needs_iron_pickaxe

    def __str__(self):
        return f"<CollectAndMineItem {self.desired_item_type} {self.desired_quantity} " \
               f"{self.relevant_block_types} {self.max_tries} success={self.success}>"

    def do(self, poly_client: poly.PolycraftInterface) -> dict:
        ''' Runs this macro action '''
        current_state = PolycraftState.create_current_state(poly_client)
        initial_quantity = current_state.count_items_of_type(self.desired_item_type)
        for i in range(self.max_tries):
            # Choose action
            action = self._choose_action(current_state)
            if action is not None:
                result = action.do(poly_client)
                current_state = PolycraftState.create_current_state(poly_client)
                new_quantity = current_state.count_items_of_type(self.desired_item_type)
                if new_quantity-initial_quantity>=self.desired_quantity:
                    self.success=True
                    return result
            else: # No action relevant - do nothing and report failure
                result = PolyNoAction().do(poly_client)
                break
        self.success = False
        return result

    def _choose_action(self, current_state):
        ''' Choose which action to try next in this macro action '''
        entity_items = current_state.get_entities_of_type(EntityType.ITEM.value)
        relevant_cells = []
        for entity_item in entity_items:
            entity_attr = current_state.entities[entity_item]
            if entity_attr["type"] == self.desired_item_type:
                cell = coordinates_to_cell(entity_attr["pos"])
                if current_state.game_map[cell]["isAccessible"]:
                    relevant_cells.append(cell)
                    break
        if len(relevant_cells) > 0:
            cell = random.choice(relevant_cells)
            action = TeleportAndFaceCell(cell)
        else:
            # Search for relevant blocks to mine
            relevant_cells = []
            for relevant_block_type in self.relevant_block_types:
                relevant_cells.extend(current_state.get_cells_of_type(relevant_block_type, only_accessible=True))
            if len(relevant_cells) > 0:
                cell = random.choice(relevant_cells)
                if self.needs_iron_pickaxe and ItemType.IRON_PICKAXE.value != current_state.get_selected_item():
                    action = PolySelectItem(ItemType.IRON_PICKAXE.value)
                else:
                    action = PolyBreakAndCollect(cell)
            else:
                logger.info(f"Can't find any blocks of the relevant types ({self.relevant_block_types}). Action failed")
                self.success = False
                action = None
        return action


class MoveToCraftingTable(PolycraftAction):
    ''' Macro action to go near a crafting table '''
    def __init__(self):
        super().__init__()

    def __str__(self):
        return f"<MoveToCraftingTable success={self.success}>"

    def do(self, poly_client: poly.PolycraftInterface) -> dict:
        state = PolycraftState.create_current_state(poly_client)
        crafting_table_cells = state.get_cells_of_type(BlockType.CRAFTING_TABLE.value, only_accessible=True)
        assert (len(crafting_table_cells)>0)
        target_cell = crafting_table_cells[0]
        action = TeleportAndFaceCell(target_cell)
        result = action.do(poly_client)
        if action.is_success(result) == False:
            self.success = False
            return result

        self.success = True
        return result

class PlaceTreeTap(PolycraftAction):
    def __init__(self, cell:str):
        super().__init__()
        self.cell = cell

    def __str__(self):
        return f"<PlaceTreeTap cell={self.cell} success={self.success}>"

    def do(self, poly_client: poly.PolycraftInterface) -> dict:
        # Place the tree tap
        action = PolyPlaceTreeTap()
        result = action.do(poly_client)
        if action.is_success(result) == False:
            self.success = False
            return result
        self.success = True
        return result

#### MACRO ACTIONS

class TeleportToAndDo(MacroAction):
    ''' Macro action that teleports to a cell if needed, turns to face it if needed, and performs an action '''
    def __init__(self, cell: str):
        super().__init__()
        self.cell = cell

    def _action_at_cell(self):
        raise NotImplementedError("Subclasses should implement this. Returns the next action to do after facing a cell. Do not forget to mark _is_done when done.")

    def _get_next_action(self)->PolycraftAction:
        state = self._current_state

        # If not near the cell, teleport to it
        if is_adjacent_to_steve(self.cell, state)==False:
            return TeleportAndFaceCell(self.cell)
        turn_angle = get_angle_to_adjacent_cell(self.cell, state)

        # If not facing the cell, turn to it
        if turn_angle != 0:
            return PolyTurn(turn_angle)

        return self._action_at_cell()

class TeleportToAndCollect(TeleportToAndDo):
    ''' Macro for teleporting to a given cell, turning to face it, and collecting an item from it.
    The macro only teleports to the cell if Steve isn't already adjacent to it, and only turns to face the cell if needed. '''
    def __init__(self, cell: str):
        super().__init__(cell)

    def __str__(self):
        return "<TeleportToAndCollect {} success={}>".format(self.cell, self.success)

    def _action_at_cell(self):
        self._is_done = True
        return PolyCollect()

class TeleportToBreakAndCollect(TeleportToAndDo):
    ''' Macro for teleporting to a given cell, turning to face it, breaking it, and collecting the resulting item.
    The macro only teleports to the cell if Steve isn't already adjacent to it, and only turns to face the cell if needed. '''
    def __init__(self, cell: str):
        super().__init__(cell)

    def __str__(self):
        return "<TeleportToBreakAndCollect {} success={}>".format(self.cell, self.success)

    def _action_at_cell(self):
        self._is_done = True
        return PolyBreakAndCollect(self.cell)

class TeleportToTableAndCraft(TeleportToAndDo):
    ''' Move to the crafting table (if not already there) and crafts according to a recipe '''
    def __init__(self, table_cell, recipe):
        super().__init__(table_cell)
        self.recipe = recipe

    def __str__(self):
        output_items = "_".join([f"{output['Item']}_{output['stackSize']}" for output in self.recipe["outputs"]])
        input_items = "_".join([f"{input['Item']}_{input['stackSize']}" for input in self.recipe["inputs"]])
        return f"<TeleportToTableAndCraft_{output_items}_from_{input_items}_at_{self.cell} success={self.success}>"

    def _action_at_cell(self):
        self._is_done = True
        return PolyCraftItem.create_action(self.recipe)

class TeleportToTraderAndTrade(TeleportToAndDo):
    ''' Move to a selected trader (if needed) and performs a trade '''
    def __init__(self, trader_id, trade):
        super().__init__(cell=None) # Cell is determined in runtime, according to the location of the trader
        self.trader_id = trader_id
        self.trade = trade

    def __str__(self):
        output_items = "_".join([f"{output['Item']}_{output['stackSize']}" for output in self.trade["outputs"]])
        input_items = "_".join([f"{input['Item']}_{input['stackSize']}" for input in self.trade["inputs"]])
        return f"<TeleportToTraderAndTrade_{self.trader_id}_{output_items}_from_{input_items} success={self.success}>"

    def _action_at_cell(self):
        self._is_done = True
        return PolyCraftItem.create_action(self.recipe)

    def _get_next_action(self)->PolycraftAction:
        state = self._current_state
        trader_obj = state.entities[self.trader_id]
        self.cell = coordinates_to_cell(trader_obj["pos"])
        return super()._get_next_action()


class CreateByRecipe(MacroAction):
    ''' An action that crafts an item by following a recipe. If ingredients are missing it goes to search and collect them '''
    def __init__(self, item_to_craft : str,
                 needs_crafting_table:bool=False,
                 needs_iron_pickaxe:bool =False):
        super().__init__()
        self.item_to_craft = item_to_craft
        self.needs_crafting_table = needs_crafting_table
        self.needs_iron_pickaxe = needs_iron_pickaxe

    def __str__(self):
        return f"<{self.name} success={self.success}>"

    def _get_action_to_collect_ingredient(self, ingredient:str, quantity:int):
        ''' An action that is used to collect the given quantity of the given ingredient '''
        raise NotImplementedError("Subclasses need to implement")

    def _get_next_action(self)->PolycraftAction:
        ''' Return an action to perform '''
        state = self._current_state
        recipe = state.get_recipe_for(self.item_to_craft)
        missing_ingredients = compute_missing_ingredients(recipe, state)

        # We have all ingredients, time to craft
        if len(missing_ingredients) == 0:
            if self.needs_crafting_table:
                if state.is_facing_type(BlockType.CRAFTING_TABLE.value)==False:
                    return MoveToCraftingTable()
            # Craft the item!
            self._is_done = True
            return PolyCraftItem.create_action(recipe)

        # Missing ingredients: go get them!

        # Equip iron pick axe if needed
        if self.needs_iron_pickaxe:
            if ItemType.IRON_PICKAXE.value != state.get_selected_item():
                return PolySelectItem(ItemType.IRON_PICKAXE.value)

        # Compute missing ingredients
        for item_needed, needed_quantity in missing_ingredients.items():
            return self._get_action_to_collect_ingredient(item_needed, needed_quantity)

class CreateByTrade(MacroAction):
    ''' An action that obtains an item by trading. If ingredients are missing it goes to search and collect it '''
    def __init__(self, item_type_to_get : str, item_types_to_give: list):
        super().__init__()
        self.item_type_to_get = item_type_to_get
        self.item_types_to_give = item_types_to_give

    def _get_action_to_collect_ingredient(self, ingredient:str, quantity:int):
        ''' An action that is used to collect the given quantity of the given ingredient '''
        raise NotImplementedError("Subclasses need to implement")

    def _get_next_action(self)->PolycraftAction:
        ''' Return an action to perform '''
        state = self._current_state
        (trader_id, trade) = state.get_trade_for(self.item_type_to_get, self.item_types_to_give)
        missing_ingredients = compute_missing_ingredients(trade, state)

        # We have all ingredients, time to craft
        if len(missing_ingredients) == 0:
            trader_cell = coordinates_to_cell(state.entities[trader_id]["pos"])
            if is_adjacent_to_steve(trader_cell, state)==False:
                return TeleportAndFaceCell(trader_cell)

            # Trade the item!
            self._is_done = True
            return PolyTradeItems.create_action(trader_id, trade)

        # Missing ingredients: go get them!
        for item_needed, needed_quantity in missing_ingredients.items():
            return self._get_action_to_collect_ingredient(item_needed, needed_quantity)

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

class CreateBlockOfPlatinum(CreateByRecipe):
    ''' Create a block of titanium by following these steps:
        1. Collect platinum
        2. Tradefor titanium
    '''
    def __init__(self):
        super().__init__(item_type_to_get=BlockType.BLOCK_OF_PLATINUM.value, item_types_to_give = [BlockType.BLOCK_OF_PLATINUM.value])

    def _get_action_to_collect_ingredient(self, ingredient:str, quantity:int):
        ''' An action that is used to collect the given quantity of the given ingredient '''
        if ingredient == BlockType.BLOCK_OF_PLATINUM.value:
            return CreateBlockOfPlatinum()
        else:
            raise ValueError(f"Unknown ingredient for {self.item_type_to_get}: {ingredient}")


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
                    return PlaceTreeTap(cell)

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



