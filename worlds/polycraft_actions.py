import math

from worlds.polycraft_interface.client import *
from worlds.polycraft_interface.client.polycraft_interface import MoveDir
from worlds.polycraft_world import *

class PolyNoAction(PolycraftAction):
    """ A no action (do nothing) """

    def __str__(self):
        return "<PolyNoAction>"

    def do(self, state:PolycraftState, env: Polycraft) -> dict:
        return env.poly_client.CHECK_COST()


class PolyTP(PolycraftAction):
    """ Teleport to a position "dist" away from the given cell (cell name is its coordinates)"""
    def __init__(self, cell:str, dist: int = 0):
        super().__init__()
        self.cell = cell
        self.dist = dist

    def __str__(self):
        return "<PolyTP pos=({}) dist={} success={}>".format(self.cell, self.dist, self.success)

    def do(self, state:PolycraftState, env: Polycraft) -> dict:
        result = env.poly_client.TP_TO_POS(self.cell, distance=self.dist)
        self.success = self.is_success(result)
        return result

class PolyMove(PolycraftAction):
    ''' Move to one of the 8 neighboring cells '''
    def __init__(self, move_dir:MoveDir, steps=1):
        super().__init__()
        self.move_dir = move_dir
        self.steps = steps

    def __str__(self):
        return f"<PolyMove move_dir={self.move_dir.name} steps={self.steps} success={self.success}>"

    def do(self, state:PolycraftState, env: Polycraft) -> dict:
        for i in range(self.steps):
            response = env.poly_client.MOVE(self.move_dir)
            if self.is_success(response)==False:
                self.success=False
                return response
        self.success = True
        return response

class PolyEntityTP(PolycraftAction):
    """ Teleport to a position "dist" away from the entity facing in direction d and with pitch p"""
    def __init__(self, entity_id: str, dist: int = 0):
        super().__init__()
        self.entity_id = entity_id
        self.dist = dist

    def __str__(self):
        return "<PolyEntityTP entity={} dist={} success={}>".format(self.entity_id, self.dist, self.success)

    def do(self, state:PolycraftState, env: Polycraft) -> dict:
        result =  env.poly_client.TP_TO_ENTITY(self.entity_id)
        self.success = self.is_success(result)
        return result


class PolyTurn(PolycraftAction):
    """ Turn the actor side to side in the y axis (vertical) in increments of 15 degrees """
    def __init__(self, direction: int):
        super().__init__()
        self.direction = direction

    def __str__(self):
        return "<PolyTurn dir={} success={}>".format(self.direction, self.success)

    def do(self, state:PolycraftState, env: Polycraft) -> dict:
        result = env.poly_client.TURN(self.direction)
        self.success = self.is_success(result)
        return result


class PolyTilt(PolycraftAction):
    """ Tilt the actor's focus up/down in the x axis (horizontal) in increments of 15 degrees """
    def __init__(self, pitch: str):
        super().__init__()
        self.pitch = pitch

    def __str__(self):
        return "<PolyTilt pitch={} success={}>".format(self.pitch, self.success)

    def do(self, state:PolycraftState, env: Polycraft) -> dict:
        result = env.poly_client.SMOOTH_TILT(self.pitch)
        self.success = self.is_success(result)
        return result


class PolyBreak(PolycraftAction):
    """ Break the block directly in front of the actor """

    def __str__(self):
        return "<PolyBreak success={}>".format(self.success)

    def do(self, state:PolycraftState, env: Polycraft) -> dict:
        result = env.poly_client.BREAK_BLOCK()
        self.success = self.is_success(result)
        return result

    def can_do(self, state:PolycraftState, env) -> bool:
        ''' Make sure no entity is occupying the space where we want to place the tree tap'''
        if state.is_facing_type(BlockType.AIR.value):
            logger.info(f"Cannot do action {self.name} because facing block of type {state.facing_block}")
            return False
        else:
            return True

class PolyInteract(PolycraftAction):
    """ Similarly to SENSE_RECIPES, this command returns the list of available trades with a particular entity (must be adjacent) """
    def __init__(self, entity_id: str):
        super().__init__()
        self.entity_id = entity_id

    def __str__(self):
        return "<PolyInteract entity={} success={}>".format(self.entity_id, self.success)

    def do(self, state:PolycraftState, env: Polycraft) -> dict:
        result = env.poly_client.INTERACT(self.entity_id)
        self.success = self.is_success(result)
        return result


class PolySense(PolycraftAction):
    """ Senses the actor's current inventory, all available blocks, recipes and entities that are in the same room as the actor """

    def __str__(self):
        return "<PolySense success={}>".format(self.success)

    def do(self, state:PolycraftState, env: Polycraft) -> dict:
        result = env.poly_client.SENSE_ALL()
        self.success = self.is_success(result)
        return result


class PolySelectItem(PolycraftAction):
    """ Select an item by name within the actor's inventory to be the item that the actor is currently holding (active item).  Pass no item name to deselect the current selected item. """
    def __init__(self, item_name: str):
        super().__init__()
        self.item_name = item_name

    def __str__(self):
        return "<PolySelectItem item={} success={}>".format(self.item_name, self.success)

    def do(self, state:PolycraftState, env: Polycraft) -> dict:
        result = env.poly_client.SELECT_ITEM(item_name=self.item_name)
        self.success = self.is_success(result)
        return result


class PolyUseItem(PolycraftAction):
    """ Perform the use action (use key on safe, open door) with the item that is currently selected.  Alternatively, pass the item in to use that item. """
    def __init__(self, item_name: str = ""):
        super().__init__()
        self.item_name = item_name

    def __str__(self):
        return "<PolyUseItem item={} success={}>".format(self.item_name, self.success)

    def do(self, state:PolycraftState, env: Polycraft) -> dict:
        result = env.poly_client.USE_ITEM(item_name=self.item_name)
        self.success = self.is_success(result)
        return result


class PolyPlaceItem(PolycraftAction):
    """ Place a block or item from the actor's inventory in the space adjacent to the block in front of the player.  This command may fail if there is no block available to place the item upon. """
    def __init__(self, item_name: str):
        super().__init__()
        self.item_name = item_name

    def __str__(self):
        return "<PolyPlaceItem item={} success={}>".format(self.item_name, self.success)

    def do(self, state:PolycraftState, env: Polycraft) -> dict:
        result = env.poly_client.PLACE(self.item_name)
        self.success = self.is_success(result)
        return result


class PolyPlaceTreeTap(PolycraftAction):
    """ Places a tree tap (polycraft:tree_tap) """
    def __init__(self):
        super().__init__()

    def do(self, state:PolycraftState, env: Polycraft) -> dict:
        result = env.poly_client.PLACE_TREE_TAP()
        self.success = self.is_success(result)
        return result

    def can_do(self, state:PolycraftState, env) -> bool:
        ''' Make sure no entity is occupying the space where we want to place the tree tap'''
        if state.is_facing_type(BlockType.AIR.value):
            return True
        else:
            logger.info(f"Cannot do action {self.name} because facing block of type {state.facing_block}")
            return False

class PolyCollect(PolycraftAction):
    """ Collect item from block in front of actor - use for collecting rubber from a tree tap. """

    def __str__(self):
        return "<PolyCollect success={}>".format(self.success)

    def do(self, state:PolycraftState, env: Polycraft) -> dict:
        result = env.poly_client.COLLECT()
        self.success = self.is_success(result)
        return result


class PolyGiveUp(PolycraftAction):
    ''' An action in which the agent gives up'''
    def __str__(self):
        return "<PolyGiveUp success={}>".format(self.success)

    def do(self, state:PolycraftState, env: Polycraft) -> dict:
        result = env.poly_client.GIVE_UP()
        self.success = self.is_success(result)
        return result


class PolyDeleteItem(PolycraftAction):
    """ Deletes the item in the player's inventory to prevent a fail state where the player is unable to pick up items due to having a full inventory """
    def __init__(self, item_name: str):
        super().__init__()
        self.item_name = item_name

    def __str__(self):
        return "<PolyDeleteItem item={} success={}>".format(self.item_name, self.success)

    def do(self, state:PolycraftState, env: Polycraft) -> dict:
        result = env.poly_client.DELETE(self.item_name)
        self.success = self.is_success(result)
        return result


class PolyTradeItems(PolycraftAction):
    """
    Perform a trade action with an adjacent entity. Accepts up to 5 items, and can result in up to 5 items.
    "items" is a list of tuples with format ( {"item_name": str, "stackSize":int} )
    """
    def __init__(self, entity_id: str, items: list):
        super().__init__()
        self.entity_id = entity_id
        self.items = items

    def __str__(self):
        return "<PolyTradeItems entity={} items={} success={}>".format(self.entity_id, self.items, self.success)

    def do(self, state:PolycraftState, env: Polycraft) -> dict:
        result = env.poly_client.TRADE(self.entity_id, self.items)
        self.success = self.is_success(result)
        return result

    @staticmethod
    def create_action(trader_id, trade_obj):
        ''' Create a PolyCraftItem action from a given recipe object '''
        trade_inputs = trade_obj['inputs']

        # Arrange trade inputs in the required format
        trade_list = []
        for recipe_item in trade_inputs:
            item_name = recipe_item['Item']
            quantity = recipe_item['stackSize']
            trade_list.append({"Item": item_name, "stackSize": quantity})

        return PolyTradeItems(entity_id=trader_id, items=trade_list)


class PolyCraftItem(PolycraftAction):
    """
    Craft an item using resources from the actor's inventory.
    [From Stephen Goss's (UTD) explanation in Slack]
    Recipes in Minecraft work in a 3x3 matrix format. So the sticks recipe is stored as [[planks, 0, 0],[planks, 0, 0],[0,0,0]].
    This means that you need sticks in slots 0 and 3.  The 2x2 crafting matrix represents a sub-matrix of the regular 3x3 grid.
    The problem is that slots only have one integer identifier, so the matrix is flattened out into an array starting with the top left slot.
    NOTE: "0" stands for a null/empty space in the matrix
    """
    def __init__(self, recipe: list):
        super().__init__()
        self.recipe = recipe

    @staticmethod
    def create_action(recipe_obj):
        ''' Create a PolyCraftItem action from a given recipe object '''
        slot_to_item = dict()
        recipe_inputs = recipe_obj['inputs']

        # Ugly special case of slot=-1 which occurs in planks. Assumption: if slot==-1 it means it doesn't matter. TODO: Verify this with UTD
        if len(recipe_inputs)==1 and recipe_inputs[0]['slot']==-1:
            adjusted_recipe_inputs = list()
            adjusted_recipe_inputs.append({'Item': recipe_inputs[0]['Item'], 'slot':0, 'stackSize':recipe_inputs[0]['stackSize']})
            recipe_inputs = adjusted_recipe_inputs

        for recipe_item in recipe_inputs:
            item_name = recipe_item['Item']
            assert (recipe_item['stackSize'] == 1)  # Currently supporting only one item per slot recipes
            slot = recipe_item['slot']
            slot_to_item[slot] = item_name

        # TODO: Better understand the behavior of this with UTD
        matrix = list()
        for i in range(3):
            row = list()
            matrix.append(row)
            for j in range(3):
                row.append("0")

        for slot, item in slot_to_item.items():
            row = math.floor(slot/3)
            col = slot % 3
            matrix[row][col]=item

        # Infer if this is a 3x3 or 2x2 recipe
        if max(slot_to_item.keys()) > 3:  # then this is  3x3 recipe
            recipe_dim = 3
        else:  # then this is a 2x2 recipe
            recipe_dim = 2
        recipe_items = []
        for row in range(recipe_dim):
            for col in range(recipe_dim):
                recipe_items.append(matrix[row][col])
        return PolyCraftItem(recipe=recipe_items)

    def __str__(self):
        return "<PolyCraftItem recipe={} success={}>".format(self.recipe, self.success)

    def do(self, state:PolycraftState, env: Polycraft) -> dict:
        result = env.poly_client.CRAFT(self.recipe)
        self.success = self.is_success(result)
        return result