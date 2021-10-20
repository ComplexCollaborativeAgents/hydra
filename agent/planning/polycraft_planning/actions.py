import random

from utils.polycraft_utils import *
from worlds.polycraft_world import *

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

        cell_coords = cell_to_coordinates(self.cell)
        steve_coords = current_state.location["pos"]
        delta = [int(cell_coords[i]) - int(steve_coords[i]) for i in range(len(cell_coords))]
        required_facing = None
        if delta == [1, 0, 0]:
            required_facing = poly.FacingDir.EAST
        elif delta == [-1, 0, 0]:
            required_facing = poly.FacingDir.WEST
        elif delta == [0, 0, 1]:
            required_facing = poly.FacingDir.SOUTH
        elif delta == [0, 0, -1]:
            required_facing = poly.FacingDir.NORTH
        else:
            raise ValueError(f'Unknown delta between cell and steve after teleport: {delta}')

        current_facing = poly.FacingDir(current_state.location["facing"])
        turn_angle = current_facing.get_angle_to(required_facing)
        if turn_angle == 0:
            return result
        else:
            turn_action = PolyTurn(turn_angle)
            result = turn_action.do(poly_client)
            if turn_action.is_success(result) == False:
                logger.info(f"teleport_and_face_cell({self.cell}) failed during TURN, Message: {result}")
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
    def __init__(self, desired_item_type: str, desired_quantity:int, relevant_block_types:list, max_tries = 5):
        super().__init__()
        self.desired_item_type = desired_item_type
        self.desired_quantity = desired_quantity
        self.relevant_block_types = relevant_block_types
        self.max_tries = max_tries # Declare failure if after max_tries iterations of collecting and mining blocks the desired quantity hasn't been reached.

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
            result = action.do(poly_client)
            current_state = PolycraftState.create_current_state(poly_client)
            new_quantity = current_state.count_items_of_type(self.desired_item_type)
            if new_quantity-initial_quantity>=self.desired_quantity:
                self.success=True
                return result

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
            # If non exists, just search for accessible blocks we can mine
            if len(relevant_cells) == 0:
                type_to_cells = current_state.get_type_to_cells()
                for type in type_to_cells:
                    if type in [BlockType.AIR.value, BlockType.BEDROCK.value]:
                        continue
                    else:
                        relevant_cells.extend(current_state.get_cells_of_type(type, only_accessible=True))

            if len(relevant_cells) > 0:
                cell = random.choice(relevant_cells)
                action = PolyBreakAndCollect(cell)
            else:
                action = PolyNoAction()
        return action


class PolyDirectCommand(PolycraftAction):
    ''' An action in which accepts a string and calls directly the polycraft agent. For debug purposes'''
    def __init__(self, command: str):
        super().__init__()
        self.command = command

    def __str__(self):
        return "<PolyDirectCommand command={} success={}>".format(self.command, self.success)

    def do(self, poly_client: poly.PolycraftInterface) -> dict:
        poly_client._send_cmd(self.command)
        result = poly_client._recv_response(self.command)
        self.success = self.is_success(result)
        return result
