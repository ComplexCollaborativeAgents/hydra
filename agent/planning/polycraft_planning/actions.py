
from worlds.polycraft_world import *
import math

class PolyBreakAndCollect(PolycraftAction):
    """ Teleport near a brick, break it, and collect the resulting item """
    def __init__(self, cell: str):
        super().__init__()
        self.cell = cell

    def __str__(self):
        return "<PolyBreakAndCollect {} success={}>".format(self.cell, self.success)

    def do(self, poly_client: poly.PolycraftInterface) -> dict:
        # Move
        result = poly_client.TP_TO_POS(self.cell, 1)
        if self.is_success(result)==False:
            self.success=False
            logger.info(f"Action {str(self)} failed during TP_TO_POS, Message: {result}")
            return result

        # Orient so we face the block
        current_state = PolycraftState.create_current_state(poly_client)
        cell_coords = self.cell.split(",")
        steve_coords = current_state.location["pos"]
        delta = [int(cell_coords[i])-int(steve_coords[i]) for i in range(len(cell_coords))]
        required_facing = None
        if delta==[1,0,0]:
            required_facing = poly.FacingDir.EAST
        elif delta==[-1,0,0]:
            required_facing = poly.FacingDir.WEST
        elif delta == [0, 0, 1]:
            required_facing = poly.FacingDir.SOUTH
        elif delta == [0, 0, -1]:
            required_facing = poly.FacingDir.NORTH
        else:
            raise ValueError(f'Unknown delta between cell and steve after teleport: {delta}')

        # TODO: Compute the right turn degree instead of this code
        while current_state.location["facing"]!=required_facing.value:
            result = poly_client.TURN(90)
            if self.is_success(result) == False:
                self.success = False
                logger.info(f"Action {str(self)} failed during TURN, Message: {result}")
                return result

        # Store the state before breaking, to identify the new item
        pre_break_state = PolycraftState.create_current_state(poly_client)

        # Break the block!
        result = poly_client.BREAK_BLOCK()
        if self.is_success(result) == False:
            self.success = False
            logger.info(f"Action {str(self)} failed during BREAK_BLOCK, Message: {result}")
            return result


        # Find and collect the item
        post_break_state = PolycraftState.create_current_state(poly_client)

        # If item in inventory - success!
        state_diff = post_break_state.diff(pre_break_state)
        if "inventory" in state_diff:
            self.success = True
            return result

        assert("entities" in state_diff) # If the new item is not in the inventory, it should be a new EntityItem
        entities_diff = state_diff["entities"]
        new_entity_items = []
        for entity_id, entity_attr in entities_diff.items():
            if entity_attr['self']['type']=='EntityItem':
                new_entity_items.append((entity_id, entity_attr['self']))

        assert(len(new_entity_items)>0)
        if len(new_entity_items)>=1: # Choose the closest entity item
            min_dist_to_item = None
            new_item_pos = None
            new_item = None
            steve_pos = post_break_state.location['pos']
            for (entity_id, entity_attr) in new_entity_items:
                item_pos = entity_attr['pos']
                if new_item_pos is None:
                    min_dist_to_item = self._distance(steve_pos, item_pos)
                    new_item_pos = item_pos
                    new_item = entity_id
                else:
                    dist_to_item = self._distance(steve_pos, item_pos)
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

        self.success = True
        return result

    def _distance(self, pos1, pos2):
        ''' Computes Euclidean distance between two vectors. TODO: Understand why math.dist() is not working in our conda environment '''
        return math.sqrt(sum([(pos1[i] - pos2[i]) ** 2 for i in range(len(pos1))]))



class PolyMoveToAndBreak(PolycraftAction):
    """ Teleport near a brick and break it """
    def __init__(self, cell: str):
        super().__init__()
        self.cell = cell

    def __str__(self):
        return "<PolyMoveToAndBreak {} success={}>".format(self.cell, self.success)

    def do(self, poly_client: poly.PolycraftInterface) -> dict:
        # Move
        result = poly_client.TP_TO_POS(self.cell, 1)
        if self.is_success(result)==False:
            self.success=False
            logger.info(f"Action {str(self)} failed, Message: {result}")
            return result

        # Orient so we face the block
        current_state = PolycraftState.create_current_state(poly_client)
        cell_coords = self.cell.split(",")
        steve_coords = current_state.location["pos"]
        delta = [int(cell_coords[i])-int(steve_coords[i]) for i in range(len(cell_coords))]
        required_facing = None
        if delta==[1,0,0]:
            required_facing = poly.FacingDir.EAST
        elif delta==[-1,0,0]:
            required_facing = poly.FacingDir.WEST
        elif delta == [0, 0, 1]:
            required_facing = poly.FacingDir.SOUTH
        elif delta == [0, 0, -1]:
            required_facing = poly.FacingDir.NORTH
        else:
            raise ValueError(f'Unknown delta between cell and steve after teleport: {delta}')

        # TODO: Compute the right turn degree instead of this code
        while current_state.location["facing"]!=required_facing.value:
            result = poly_client.TURN(90)
            if self.is_success(result) == False:
                self.success = False
                logger.info(f"Action {str(self)} failed, Message: {result}")
                return result

        # Break the block!
        result = poly_client.BREAK_BLOCK()
        self.success = self.is_success(result)
        return result



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
