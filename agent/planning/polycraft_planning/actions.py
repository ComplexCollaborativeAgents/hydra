import copy
import random

from utils.polycraft_utils import *
from worlds.polycraft_actions import *
from worlds.polycraft_world import *

class MacroAction(PolycraftAction):
    ''' A macro action is a generator of basic PolycraftActions based on the current state '''
    def __init__(self, max_steps:int=1):
        super().__init__()
        self.actions_done = [] # a list of the actions performed in this macro action. Useful for debugging.
        self._is_done = False
        self.max_steps = max_steps # Maximal number of steps (actions) required to do this macro action is expected

    def _get_next_action(self, state:PolycraftState, env: Polycraft)->PolycraftAction:
        raise NotImplementedError("Subclasses of MacroAction should implement this and set self.next_action in it")

    def is_done(self):
        return self._is_done

    def do(self, state:PolycraftState, env: Polycraft) -> dict:
        ''' Key: macro action accept the environment, not the polycraft_interface'''
        logger.info(f"Doing macro action {self} until done")
        result = None
        i = 0
        while i<self.max_steps:
            next_action = self._get_next_action(state, env)
            next_state = state
            if next_action is not None:
                next_state, step_cost = env.act(state, next_action)
                result = next_action.response
                self.actions_done.append(next_action)
            if self.is_done():
                return result
            i = i+1
            state = next_state
        return result


class WaitForLogs(MacroAction):
    ''' Wait a predefined number of steps or until we see blocks of type log '''
    def __init__(self):
        super().__init__(max_steps = 10)

    def _get_next_action(self, state:PolycraftState, env: Polycraft)->PolycraftAction:
        if len(state.get_cells_of_type(BlockType.LOG.value, only_accessible=False))>0:
            self._is_done = True
        return PolyNoAction()

class BreakAndCollect(MacroAction):
    MAX_COLLECT_RANGE_AFTER_BREAK = 4
    MAX_STEPS = 4**MAX_COLLECT_RANGE_AFTER_BREAK

    """ Teleport near a brick, break it, and collect the resulting item """
    def __init__(self, cell: str):
        super().__init__(max_steps=BreakAndCollect.MAX_STEPS)
        self.cell = cell
        self.state_before_break = None
        self.entity_items_to_collect = None
        self.break_action = None # Store the break action, since its result and success are the one we consider

    def __str__(self):
        return "<BreakAndCollect {} success={}>".format(self.cell, self.success)

    def do(self, state:PolycraftState, env: Polycraft) -> dict:
        ''' Key: macro action accept the environment, not the polycraft_interface'''
        result = super().do(state, env)
        break_result = self.break_action.response
        self.success = self.break_action.is_success(break_result)
        if result is not None:
            return break_result

    def can_do(self, state:PolycraftState, env) -> bool:
        ''' Make sure no entity is occupying the space where we want to place the tree tap'''
        if state.is_facing_type(BlockType.AIR.value):
            logger.info(f"Cannot do action {self.name} because facing block of type {state.facing_block}")
            return False
        else:
            return True

    def _get_next_action(self, state:PolycraftState, env: Polycraft)->PolycraftAction:
        if self.state_before_break is None:
            self.state_before_break=copy.deepcopy(state)
            self.break_action = PolyBreak()  # Store break action to correctly return the result of this macro action (collect after break sometimes fail even when break and collect has worked)
            return self.break_action

        # Compute entities to collect
        if self.entity_items_to_collect is None:
            state_diff = state.diff(self.state_before_break)
            self.entity_items_to_collect = get_new_entity_items(state_diff)
        self.entity_items_to_collect = self._get_entity_items_to_collect(current_state=state,
                                                                         max_range=BreakAndCollect.MAX_COLLECT_RANGE_AFTER_BREAK)
        if len(self.entity_items_to_collect)==0:
            self._is_done = True
            return None
        else: # len(self.entity_items_to_collect)>0:
            # Search for the mined items. Some may be in invenotry, some in the near by area as EntityItem objects
            entity_id = self.entity_items_to_collect.pop(0)
            entity_cell = coordinates_to_cell(state.entities[entity_id]["pos"])
            return PolyEntityTP(entity_id)

    def _get_entity_items_to_collect(self, current_state, max_range):
        ''' Returns a list on entity items to collect after the break '''
        if self.entity_items_to_collect is None: # consider entity items that have appeared after breaking the block
            return get_new_entity_items(current_state.diff(self.state_before_break))

        block_coord = cell_to_coordinates(self.cell)
        relevant_entities = []
        min_dist_to_entity = None
        for entity_id in self.entity_items_to_collect:
            if entity_id not in current_state.entities:
                continue
            item_pos = current_state.entities[entity_id]['pos']
            dist_to_break_pos = compute_cell_distance(block_coord, item_pos)
            if dist_to_break_pos<=max_range:
                if min_dist_to_entity is None:
                    min_dist_to_entity = dist_to_break_pos
                    relevant_entities.append(entity_id)
                elif min_dist_to_entity > dist_to_break_pos:
                    min_dist_to_entity = dist_to_break_pos
                    relevant_entities.insert(0,entity_id)
                else:
                    relevant_entities.append(entity_id)
        return relevant_entities

class TeleportAndFaceCell(MacroAction):
    MAX_STEPS = 6 # TODO: Rethink this

    ''' Macro for teleporting to a given cell and turning to face it '''
    def __init__(self, cell: str):
        super().__init__(max_steps=TeleportAndFaceCell.MAX_STEPS)
        self.cell = cell

    def __str__(self):
        return "<TeleportAndFaceCell {} success={}>".format(self.cell, self.success)

    def _get_next_action(self, state:PolycraftState, env: Polycraft)->PolycraftAction:
        # If not in the same room at Steve, move to that room
        path_to_room = get_door_path_to_cell(self.cell, state)
        if len(path_to_room)>0:
            # Need to move through some doors to get to the cell
            return MoveThroughDoor(path_to_room[0])

        # If not near the cell - teleport to it
        if is_adjacent_to_steve(self.cell, state)==False:
            return PolyTP(self.cell, dist=1)

        # If not facing the cell, turn to face it
        self._is_done = True
        turn_angle = get_angle_to_adjacent_cell(self.cell, state)
        if turn_angle == 0:
            return None # TODO: Add a mechanism that says no action needed. Not critical to do this
        else:
            return PolyTurn(turn_angle)


#### MACRO ACTIONS

class TeleportToAndDo(MacroAction):
    ''' Macro action that teleports to a cell if needed, turns to face it if needed, and performs an action '''
    def __init__(self, cell: str, max_steps:int):
        super().__init__(max_steps=max_steps)
        self.cell = cell

    def _action_at_cell(self, state:PolycraftState):
        raise NotImplementedError("Subclasses should implement this. Returns the next action to do after facing a cell. Do not forget to mark _is_done when done.")

    def _get_next_action(self, state:PolycraftState, env: Polycraft)->PolycraftAction:
        # If not near the cell, teleport to it
        if is_adjacent_to_steve(self.cell, state)==False:
            return TeleportAndFaceCell(self.cell)
        turn_angle = get_angle_to_adjacent_cell(self.cell, state)

        # If not facing the cell, turn to it
        if turn_angle != 0:
            return PolyTurn(turn_angle)

        return self._action_at_cell(state)

class ExploreRoom(PolycraftAction):
    ''' This action updates the environment (Polycraft) by sensing the current room. '''

    def __init__(self, door_cell:str):
        ''' door_cell is the cell of the door though which we entered the room. Important for remmebering how to enter/exist the room to reach the cells in it '''
        super().__init__()
        self.door_cell = door_cell

    def do(self, state: PolycraftState, env: Polycraft):
        result = env.poly_client.SENSE_ALL()

        # Update game map knowledge with the sensed knowledge (note: sense only returns the game map for the current room)
        sensed_game_map = result['map']
        new_room_explored = False
        for cell_id, cell_attr in sensed_game_map.items():
            known_cell = False
            for door_cell_id, room_game_map in env.door_to_room_cells.items():
                if cell_id in room_game_map:
                    known_cell = True
                    room_game_map[cell_id] = cell_attr
            if known_cell == False: #
                new_room_explored = True
            env.door_to_room_cells[self.door_cell][cell_id] = cell_attr # A cell may exists in multiple door-to-cells entries

        if new_room_explored:
            logger.info(f"Explored a new room! room reachable through door {self.door_cell}")
        return result

class MoveThroughDoor(MacroAction):
    MAX_STEPS = 3 # Maximum forward steps until explored room
    ''' Action that moves through a door '''
    def __init__(self, door_cell: str):
        super().__init__(max_steps=TeleportAndFaceCell.MAX_STEPS+MoveThroughDoor.MAX_STEPS)
        self.door_cell = door_cell
        self.before_entering = True

    def __str__(self):
        return f"<MoveThroughDoor {self.door_cell} success={self.success}>"

    def _get_next_action(self, state:PolycraftState, env: Polycraft)->PolycraftAction:
        # If not near the cell, teleport to it
        if self.before_entering:
            if is_adjacent_to_steve(self.door_cell, state)==False:
                return TeleportAndFaceCell(self.door_cell)
            turn_angle = get_angle_to_adjacent_cell(self.door_cell, state)

            # If not facing the cell, turn to it
            if turn_angle != 0:
                return PolyTurn(turn_angle)
            else:
                self.before_entering = False
                return PolyMove(MoveDir.FORWARD, MoveThroughDoor.MAX_STEPS)
        else:
            self._is_done = True
            return ExploreRoom(self.door_cell)


class TeleportToAndCollect(TeleportToAndDo):
    ''' Macro for teleporting to a given cell, turning to face it, and collecting an item from it.
    The macro only teleports to the cell if Steve isn't already adjacent to it, and only turns to face the cell if needed. '''
    def __init__(self, cell: str):
        super().__init__(cell, max_steps=TeleportAndFaceCell.MAX_STEPS+1)

    def __str__(self):
        return "<TeleportToAndCollect {} success={}>".format(self.cell, self.success)

    def _action_at_cell(self, state:PolycraftState):
        self._is_done = True
        return PolyCollect()

class TeleportToAndUse(TeleportToAndDo):
    ''' Macro for teleporting to a given cell, turning to face it, and collecting an item from it.
    The macro only teleports to the cell if Steve isn't already adjacent to it, and only turns to face the cell if needed. '''
    def __init__(self, cell: str, item_to_use:str = None):
        super().__init__(cell, max_steps=TeleportAndFaceCell.MAX_STEPS+2)
        self.item_to_use = item_to_use

    def __str__(self):
        return f"<TeleportToAndUse {self.cell} {self.item_to_use} success={self.success}>"

    def _action_at_cell(self, state:PolycraftState):
        if self.item_to_use is None or state.get_selected_item()==self.item_to_use:
            self._is_done = True
            return PolyUseItem(self.item_to_use)
        else:
            return PolySelectItem(self.item_to_use)

class TeleportToAndPlaceTreeTap(TeleportToAndDo):
    ''' Macro for teleporting to a given cell, turning to face it, and placing a tree tap on it. '''
    def __init__(self, cell: str):
        super().__init__(cell, max_steps=TeleportAndFaceCell.MAX_STEPS+2)

    def __str__(self):
        return f"<TeleportToAndPlaceTreeTap {self.cell} success={self.success}>"

    def _action_at_cell(self, state:PolycraftState):
        self._is_done = True
        return PolyPlaceTreeTap()

class TeleportToBreakAndCollect(TeleportToAndDo):
    ''' Macro for teleporting to a given cell, turning to face it, breaking it, and collecting the resulting item.
    The macro only teleports to the cell if Steve isn't already adjacent to it, and only turns to face the cell if needed. '''
    def __init__(self, cell: str):
        super().__init__(cell, max_steps=TeleportAndFaceCell.MAX_STEPS+BreakAndCollect.MAX_STEPS)

    def __str__(self):
        return "<TeleportToBreakAndCollect {} success={}>".format(self.cell, self.success)

    def _action_at_cell(self, state:PolycraftState):
        self._is_done = True
        return BreakAndCollect(self.cell)

class TeleportToTableAndCraft(TeleportToAndDo):
    ''' Move to the crafting table (if not already there) and crafts according to a recipe '''
    def __init__(self, table_cell, recipe):
        super().__init__(table_cell,max_steps=TeleportAndFaceCell.MAX_STEPS+1)
        self.recipe = recipe

    def __str__(self):
        output_items = "_".join([f"{output['Item']}_{output['stackSize']}" for output in self.recipe["outputs"]])
        input_items = "_".join([f"{input['Item']}_{input['stackSize']}" for input in self.recipe["inputs"]])
        return f"<TeleportToTableAndCraft_{output_items}_from_{input_items}_at_{self.cell} success={self.success}>"

    def _action_at_cell(self, state:PolycraftState):
        self._is_done = True
        return PolyCraftItem.create_action(self.recipe)

class TeleportToTraderAndTrade(TeleportToAndDo):
    ''' Move to a selected trader (if needed) and performs a trade '''
    def __init__(self, trader_id, trade):
        super().__init__(cell=None, max_steps=TeleportAndFaceCell.MAX_STEPS+1) # Cell is determined in runtime, according to the location of the trader
        self.trader_id = trader_id
        self.trade = trade

    def __str__(self):
        output_items = "_".join([f"{output['Item']}_{output['stackSize']}" for output in self.trade["outputs"]])
        input_items = "_".join([f"{input['Item']}_{input['stackSize']}" for input in self.trade["inputs"]])
        return f"<TeleportToTraderAndTrade_{self.trader_id}_{output_items}_from_{input_items} success={self.success}>"

    def _action_at_cell(self, state:PolycraftState):
        self._is_done = True
        return PolyTradeItems.create_action(self.trader_id, self.trade)

    def _get_next_action(self, state:PolycraftState, env: Polycraft)->PolycraftAction:
        trader_obj = state.entities[self.trader_id]
        self.cell = coordinates_to_cell(trader_obj["pos"])
        return super()._get_next_action(state, env)


class CreateByRecipe(MacroAction):
    ''' An action that crafts an item by following a recipe. If ingredients are missing it goes to search and collect them '''
    def __init__(self, item_to_craft : str,
                 needs_crafting_table:bool=False,
                 needs_iron_pickaxe:bool =False):
        super().__init__(max_steps=20)   # TODO: 20 here is a magic number. Sine this action is not used by our main agent, it's good enough for now.
        self.item_to_craft = item_to_craft
        self.needs_crafting_table = needs_crafting_table
        self.needs_iron_pickaxe = needs_iron_pickaxe

    def __str__(self):
        return f"<{self.name} success={self.success}>"

    def _get_action_to_collect_ingredient(self, ingredient:str, quantity:int):
        ''' An action that is used to collect the given quantity of the given ingredient '''
        raise NotImplementedError("Subclasses need to implement")

    def _get_next_action(self, state:PolycraftState, env: Polycraft)->PolycraftAction:
        ''' Return an action to perform '''
        recipe = state.get_recipe_for(self.item_to_craft)
        missing_ingredients = compute_missing_ingredients(recipe, state)

        # We have all ingredients, time to craft
        if len(missing_ingredients) == 0:
            if self.needs_crafting_table:
                if state.is_facing_type(BlockType.CRAFTING_TABLE.value)==False:
                    crafting_table_cell = self._find_crafting_table(state)
                    return TeleportAndFaceCell(crafting_table_cell)
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

    def _find_crafting_table(self, state:PolycraftState):
        ''' Finds a crafting table cell that is accessible. Returns None if none exists '''
        crafting_table_cells = state.get_cells_of_type(BlockType.CRAFTING_TABLE.value, only_accessible=True)
        if len(crafting_table_cells)==0:
            return None
        else:
            return random.choice(crafting_table_cells)

class CreateByTrade(MacroAction):
    ''' An action that obtains an item by trading. If ingredients are missing it goes to search and collect it '''
    def __init__(self, item_type_to_get : str, item_types_to_give: list):
        super().__init__(max_steps=20) # TODO: 20 here is a magic number. Sine this action is not used by our main agent, it's good enough for now.
        self.item_type_to_get = item_type_to_get
        self.item_types_to_give = item_types_to_give

    def _get_action_to_collect_ingredient(self, ingredient:str, quantity:int):
        ''' An action that is used to collect the given quantity of the given ingredient '''
        raise NotImplementedError("Subclasses need to implement")

    def _get_next_action(self, state:PolycraftState, env: Polycraft)->PolycraftAction:
        ''' Return an action to perform '''
        (trader_id, trade) = state.get_trade_for(self.item_type_to_get, self.item_types_to_give)
        missing_ingredients = compute_missing_ingredients(trade, state)

        # We have all ingredients, time to craft
        if len(missing_ingredients) == 0:
            try:
                trader_pos = state.entities[trader_id]["pos"]
            except KeyError as err:
                raise KeyError("Trader {} not in entities? {}".format(trader_id, state.entities))

            trader_cell = coordinates_to_cell(trader_pos)
            if is_adjacent_to_steve(trader_cell, state)==False:
                return TeleportAndFaceCell(trader_cell)

            # Trade the item!
            self._is_done = True
            return PolyTradeItems.create_action(trader_id, trade)

        # Missing ingredients: go get them!
        for item_needed, needed_quantity in missing_ingredients.items():
            return self._get_action_to_collect_ingredient(item_needed, needed_quantity)


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


################# Exploratory Actions ####################

# Break an unknown block type
# Interact with an unknown entity type
# Reach an inaccessible entity
# Open chest



class OpenDoor(TeleportToAndDo):
    ''' Move to a door, open it, and explore the room '''
    def __init__(self, door_cell:str):
        super().__init__(door_cell, max_steps=TeleportAndFaceCell.MAX_STEPS+1)

    def __str__(self):
        return f"<OpenDoor_{self.cell} success={self.success}>"

    def _action_at_cell(self, state:PolycraftState):
        door_attr = state.game_map[self.cell]
        if door_attr["open"].upper()=="False".upper():
            self._is_done = True
            return PolyUseItem()

class OpenSafeAndCollect(TeleportToAndDo):
    ''' Move to a door, open it, and explore the room '''

    def __init__(self, safe_cell: str):
        super().__init__(safe_cell, max_steps=TeleportAndFaceCell.MAX_STEPS + 3)
        self.safe_opened = False

    def __str__(self):
        return f"<OpenSafeAndCollect{self.cell} success={self.success}>"

    def _action_at_cell(self, state: PolycraftState):
        known_cells = state.get_known_cells()
        if self.safe_opened==False:
            if state.get_selected_item()!=ItemType.KEY.value:
                return PolySelectItem(ItemType.KEY.value)
            else:
                self.safe_opened=True
                return PolyUseItem(ItemType.KEY.value)
        else:
            self._is_done = True
            return PolyCollect()
