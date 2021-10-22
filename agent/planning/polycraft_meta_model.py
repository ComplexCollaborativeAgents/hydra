import copy
import math
import settings
import logging
import random
from agent.planning.meta_model import *
from agent.planning.polycraft_planning.actions import *
from worlds.polycraft_world import *
from agent.planning.pddl_plus import *
logging.basicConfig(format='%(name)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("polycraft_meta_model")
logger.setLevel(logging.INFO)

class PddlObjectType():
    ''' A generator for Pddl Objects '''

    def __init__(self):
        self.hyper_parameters = dict()
        self.pddl_type="object" # This the PDDL+ type of this object.

    def _compute_obj_attributes(self, obj, problem_params: dict):
        obj_attributes = self._compute_observable_obj_attributes(obj, problem_params)
        return obj_attributes

    def _compute_observable_obj_attributes(self, obj, problem_params:dict):
        ''' Subclasses should override this setting all attributes of that object that can be observed'''
        return dict()

    def add_object_to_problem(self, prob: PddlPlusProblem, obj, problem_params:dict):
        ''' Populate a PDDL+ problem with details about this object '''

        name = self._get_name(obj)
        prob.objects.append([name, self.pddl_type])
        attributes = self._compute_obj_attributes(obj, problem_params)
        for attribute in attributes:
            value = attributes[attribute]
            # If attribute is Boolean no need for an "=" sign
            if isinstance(value, bool):
                if value==True:
                    prob.init.append([attribute, name])
                else: # value == False
                    prob.init.append(['not', [attribute, name]])
            else: # Attribute is a number
                prob.init.append(['=', [attribute, name], value])

    def add_object_to_state(self, pddl_state: PddlPlusState, obj, state_params:dict):
        ''' Populate a PDDL+ state with details about this object '''

        name = self._get_name(obj)
        attributes = self._compute_observable_obj_attributes(obj, state_params)
        for attribute in attributes:
            value = attributes[attribute]
            fluent_name = (attribute, name)
            # If attribute is Boolean no need for an "=" sign
            if isinstance(value,  bool):
                if value==True:
                    pddl_state.boolean_fluents.add(fluent_name)
                # TODO: Think how to handle booean fluents with False value. Not as trivial as it sounds
            else: # Attribute is a number
                pddl_state.numeric_fluents[fluent_name]=value

    def _get_name(self, obj):
        return '{}_{}'.format(type, obj)


# Inventory items

class PddlInventoryItemType(PddlObjectType):
    def __init__(self, block_type=-1):
        super().__init__()
        self.pddl_type = "inventory_item"
        self.block_type = block_type

    def _compute_observable_obj_attributes(self, obj, problem_params: dict):
        obj_attributes = dict()
        (item_id, item_attr) = obj
        for attr_name, attr_value in item_attr.items():
            if attr_name != "item":
                if type(attr_value)==bool or ( type(attr_value)==str and attr_value.lower() in ["true", "false"] ):
                    obj_attributes[attr_name]=bool(attr_value)
                else:
                    obj_attributes[attr_name]=attr_value

        if self.pddl_type != -1:
            obj_attributes["block_type"] = self.block_type

        return obj_attributes

    def _get_name(self, obj):
        (item_id, item_attr) = obj
        return "inventory_{}".format(item_id, item_attr["item"])

#### Cell Types

class PddlGameMapCellType(PddlObjectType):
    IGNORED_CELL_ATTRIBUTES = ["facing", "half", "hinge"] # List of attributes to not include in the PDDL model TODO: Think about this

    def __init__(self,block_type=-1):
        super().__init__()
        self.pddl_type = "cell"
        self.block_type = block_type

    def _compute_observable_obj_attributes(self, obj, problem_params:dict):
        obj_attributes = dict()
        (cell_id, cell_attr)=obj

        for attr_name, attr_value in cell_attr.items():
            if attr_name != 'name' and attr_name not in PddlGameMapCellType.IGNORED_CELL_ATTRIBUTES:
                if type(attr_value)== bool or attr_value.lower() in ["true", "false"]:
                    obj_attributes[attr_name]=bool(attr_value)
                else:
                    obj_attributes[attr_name]=attr_value

        x, y, z = cell_id.split(",")
        obj_attributes["cell_x"] = int(x)
        obj_attributes["cell_y"] = int(y)
        obj_attributes["cell_z"] = int(z)

        if self.block_type != -1:
            obj_attributes["block_type"] = self.block_type

        return obj_attributes

    def _get_name(self, obj):
        (cell_id, cell_attr) = obj
        return "cell_{}".format("_".join(cell_id.split(",")))

class PddlLogType(PddlGameMapCellType):
    def __init__(self, block_type):
        super().__init__(block_type)

    def _compute_observable_obj_attributes(self, obj, problem_params:dict):
        obj_attributes = super()._compute_observable_obj_attributes(obj, problem_params)
        # Log has a variant attribute that is not numeric and needs special treatment

        if "variant" in obj_attributes:
            if obj_attributes["variant"]=="oak":
                obj_attributes["is_oak"]=True
            obj_attributes.pop("variant")

        if "axis" in obj_attributes:
            axis_value = obj_attributes["axis"]
            known_axis_values = {"x":0, "y":1, "z":2}
            if axis_value in known_axis_values:
                obj_attributes["log_axis"]=known_axis_values[axis_value]
            obj_attributes.pop("axis")

        return obj_attributes

class PddlAirType(PddlGameMapCellType):
    def __init__(self, block_type):
        super().__init__(block_type)

    def _compute_observable_obj_attributes(self, obj, problem_params:dict):
        return dict() # Ignoring air blocks in the PDDL

# Entity types

class PddlEntityType(PddlObjectType):
    IGNORED_ENTITY_ATTRIBUTES = ['equipment', 'pos', 'name']
    def __init__(self,entity_type=-1):
        super().__init__()
        self.pddl_type = "entity"
        self.entity_type = entity_type

    def _compute_observable_obj_attributes(self, obj, problem_params:dict):
        obj_attributes = dict()
        obj_attributes["entity_type"] = self.entity_type

        (entity_id, entity_attr)=obj

        for attr_name, attr_value in entity_attr.items():
            if attr_name != 'type' and attr_name not in PddlEntityType.IGNORED_ENTITY_ATTRIBUTES:
                if type(attr_value)== bool or (type(attr_value)==str and attr_value.lower() in ["true", "false"]):
                    obj_attributes[attr_name]=bool(attr_value)
                else:
                    obj_attributes[attr_name]=attr_value

        x, y, z = entity_attr['pos']
        obj_attributes["entity_x"] = x
        obj_attributes["entity_y"] = y
        obj_attributes["entity_z"] = z


        return obj_attributes

    def _get_name(self, obj):
        (entity_id, entity_attr)=obj
        return "entity_{}".format(entity_id)

class PddlEntityItemType(PddlEntityType):
    ITEM_TO_ENUM = {"minecraft:sapling":1}

    def __init__(self,entity_type=-1):
        super().__init__(entity_type)

    def _compute_observable_obj_attributes(self, obj, problem_params:dict):
        obj_attributes = super()._compute_observable_obj_attributes(obj, problem_params)
        obj_attributes.pop('damage') # Not supported yet
        obj_attributes.pop('maxdamage') # Not supported yet
        obj_attributes['item'] = PddlEntityItemType.ITEM_TO_ENUM[obj_attributes['item']] # Convert item to enum

        return obj_attributes

class PolycraftMetaModel(MetaModel):

    ''' Sets the default meta-model'''
    def __init__(self):
        super().__init__(docker_path=settings.POLYCRAFT_PLANNING_DOCKER_PATH,
                         domain_file_name="polycraft_domain.pddl",
                         delta_t=settings.POLYCRAFT_DELTA_T,
                         metric = 'minimize(total-time)',
                         repairable_constants=[],
                         constant_numeric_fluents={},
                         constant_boolean_fluents={})

        self.domain_name = "polycraft" # TODO: Move this to constructor
        self.problem_name = "polycraft_prob"  # TODO: Move this to constructor

        # Mapping of type to Pddl object. All objects of this type will be clones of this pddl object
        self.type_to_int = dict()
        type_index = 0
        for block_type in BlockType:
            self.type_to_int[block_type.value]=type_index
            type_index = type_index+1
        for item_type in ItemType:
            self.type_to_int[item_type.value]=type_index
            type_index = type_index+1
        for entity_type in EntityType:
            self.type_to_int[entity_type.value]=type_index
            type_index = type_index+1

        self.object_types = dict()

        self.object_types[BlockType.AIR.value]=PddlAirType(0)
        self.object_types[BlockType.BEDROCK.value]=PddlGameMapCellType(1)
        self.object_types[BlockType.LOG.value]=PddlLogType(2) # Variant
        self.object_types[BlockType.DIAMOND_ORE.value]=PddlGameMapCellType(3)
        self.object_types[BlockType.PLASTIC_CHEST.value]=PddlGameMapCellType(4) # Facing
        self.object_types[ItemType.IRON_PICKAXE.value]=PddlInventoryItemType(5)
        self.object_types[BlockType.CRAFTING_TABLE.value]=PddlGameMapCellType(6)
        self.object_types[BlockType.WOODER_DOOR.value] = PddlGameMapCellType(7)
        self.object_types[BlockType.BLOCK_OF_PLATINUM.value] = PddlGameMapCellType(8)
        self.object_types[ItemType.TREE_TAP.value] = PddlGameMapCellType(9)
        self.object_types[ItemType.PLANKS.value] = PddlInventoryItemType(10)
        self.object_types["minecraft:stick"] = PddlInventoryItemType(11)
        self.object_types["polycraft:wooden_pogo_stick"] = PddlInventoryItemType(12)
        self.object_types["polycraft:block_of_titanium"] = PddlInventoryItemType(13)
        self.object_types[ItemType.DIAMOND_BLOCK.value] = PddlInventoryItemType(14)
        self.object_types[ItemType.SACK_POLYISOPRENE_PELLETS.value] = PddlInventoryItemType(15)
        self.object_types[ItemType.DIAMOND.value] = PddlInventoryItemType(16)
        self.object_types[EntityType.TRADER.value] = PddlEntityType(17)
        self.object_types[EntityType.POGOIST.value] = PddlEntityType(18)
        self.object_types[EntityType.ITEM.value] = PddlEntityItemType(19)


        self.block_to_item = dict() # Maps the expected outcome of mining a given block type

    def create_pddl_domain(self, world_state:PolycraftState) -> PddlPlusDomain:
        ''' Create a PDDL+ domain for the given observed state '''
        domain_file = "{}/{}".format(str(self.docker_path), "polycraft_domain_template.pddl")
        domain_parser = PddlDomainParser()
        pddl_domain = domain_parser.parse_pddl_domain(domain_file)

        # Add actions for recipes
        self._add_do_recipe_actions(pddl_domain, world_state)

        # Add actions for trades
        self._add_do_trade_actions(pddl_domain, world_state)

        return pddl_domain

    def _add_do_recipe_actions(self, pddl_domain, world_state):
        ''' Add a do_recipe action for every recipe in the world state '''
        for recipe_id, recipe in enumerate(world_state.recipes):
            recipe_action = PddlPlusWorldChange(WorldChangeTypes.action)
            recipe_action.name = f'do_recipe_{recipe_id}'

            # Parameters
            param_list = []
            for input_item in recipe['inputs']:
                param_list.extend([f'?slot{input_item["slot"]}', '-', 'inventory_item'])
            recipe_action.parameters.append(param_list)

            # Effects
            assert (len(recipe['outputs']) == 1)  # Assuming a recipe creates a single item
            output = recipe['outputs'][0]
            stack_size = output["stackSize"]
            item_type = self.object_types[output["Item"]].block_type
            recipe_action.effects.append(['assign', ['crafted_item'], item_type])
            recipe_action.effects.append(['assign', ['crafted_item_count'], stack_size])

            # Preconditions
            for input_item in recipe['inputs']:
                slot = input_item["slot"]
                stack_size = input_item["stackSize"]
                item_type = self.object_types[input_item["Item"]].block_type
                recipe_action.preconditions.append(["=", ['item_type', f'?slot{slot}'], item_type])
                recipe_action.preconditions.append([">=", ['count', f'?slot{slot}'], stack_size])

                recipe_action.effects.append(['decrease', ['count', f'?slot{slot}'], stack_size])

            pddl_domain.actions.append(recipe_action)

    def _add_do_trade_actions(self, pddl_domain, world_state):
        ''' Add a do_recipe action for every recipe in the world state '''
        for trader, trades in world_state.trades.items():
            for trade_id, trade in enumerate(trades):
                recipe_action = PddlPlusWorldChange(WorldChangeTypes.action)
                recipe_action.name = f'do_trade_{trader}_{trade_id}'

                # Parameters
                param_list = []
                for input_item in trade['inputs']:
                    param_list.extend([f'?slot{input_item["slot"]}', '-', 'inventory_item'])
                recipe_action.parameters.append(param_list)

                # Effects
                assert (len(trade['outputs']) == 1)  # Assuming a recipe creates a single item
                output = trade['outputs'][0]
                stack_size = output["stackSize"]
                item_type = self.object_types[output["Item"]].block_type
                recipe_action.effects.append(['assign', ['traded_item'], item_type])
                recipe_action.effects.append(['assign', ['traded_item_count'], stack_size])

                # Preconditions
                for input_item in trade['inputs']:
                    slot = input_item["slot"]
                    stack_size = input_item["stackSize"]
                    item_type = self.object_types[input_item["Item"]].block_type
                    recipe_action.preconditions.append(["=", ['item_type', f'?slot{slot}'], item_type])
                    recipe_action.preconditions.append([">=", ['count', f'?slot{slot}'], stack_size])

                    recipe_action.effects.append(['decrease', ['count', f'?slot{slot}'], stack_size])

                pddl_domain.actions.append(recipe_action)

    def _get_steve_attributes(self, steve_obj):
        ''' Extract Steve's attributes fromthe relevant state object '''
        obj_attributes = dict()
        obj_attributes["steve_x"] = steve_obj['pos'][0]
        obj_attributes["steve_y"] = steve_obj['pos'][1]
        obj_attributes["steve_z"] = steve_obj['pos'][2]
        # obj_attributes["steve_facing"] = steve_obj["facing"] # Need to create an enum for the directions (NORTH, ...)
        obj_attributes["steve_yaw"] = steve_obj["yaw"]
        obj_attributes["steve_pitch"] = steve_obj["pitch"]
        return obj_attributes

    def create_pddl_problem(self, world_state : PolycraftState):
        ''' Creates a PDDL problem file in which the given world state is the initial state '''

        pddl_problem = PddlPlusProblem()
        pddl_problem.domain = self.domain_name
        pddl_problem.name = self.problem_name
        pddl_problem.metric = self.metric
        pddl_problem.objects = []
        pddl_problem.init = []
        pddl_problem.goal = []

        # A dictionary with global problem parameters
        problem_params = dict()

        # Add Steve location to init state
        steve_attributes = self._get_steve_attributes(world_state.location)
        for attr_name, attr_value in steve_attributes.items():
            pddl_problem.init.append(['=', [attr_name], attr_value])
        pddl_problem.init.append(['=', ['facing_block'], self.object_types[world_state.facing_block['name']].block_type])

        # Add game map cells
        for cell, cell_attr in world_state.game_map.items():
            type_str = cell_attr['name']

            if type_str not in self.object_types:
                logger.info("Unknown game map cell type: %s" % type_str)
                type = PddlGameMapCellType()
            else:
                type = self.object_types[type_str]

            type.add_object_to_problem(pddl_problem, (cell, cell_attr), problem_params)


        # Add inventory items
        for item_id, item_attr in world_state.inventory.items():
            type_str = item_attr['item']
            if len(type_str) == 0:  # Empty inventory slot
                continue
            if type_str not in self.object_types:
                logger.info("Unknown inventory object type: %s" % type_str)
                type = PddlInventoryItemType()
            else:
                type = self.object_types[type_str]
            type.add_object_to_problem(pddl_problem, (item_id, item_attr), problem_params)

        # Add other entities
        for entity, entity_attr in world_state.entities.items():
            type_str = entity_attr["type"]
            type = self.object_types[type_str]
            type.add_object_to_problem(pddl_problem, (entity, entity_attr), problem_params)

        # Add goal
        pddl_problem.goal.append(['pogo_created',])

        return pddl_problem

    def create_pddl_state(self, world_state: PolycraftState) -> PddlPlusState:
        ''' Translate the given observed world state to a PddlPlusState object '''

        pddl_state = PddlPlusState()

        # A dictionary with current state parameters that are not object related
        state_params = dict()

        # Add Steve's attributes
        steve_attributes = self._get_steve_attributes(world_state.location)
        for attr_name, attr_value in steve_attributes.items():
            pddl_state.numeric_fluents[attr_name]=attr_value
        pddl_state.numeric_fluents['facing_block'] = self.object_types[world_state.facing_block['name']].block_type

        # Add game map cells
        for cell, cell_attr in world_state.game_map.items():
            type_str = cell_attr['name']

            if type_str not in self.object_types:
                logger.info("Unknown game map cell type: %s" % type_str)
                type = PddlGameMapCellType()
            else:
                type = self.object_types[type_str]

            type.add_object_to_state(pddl_state, (cell, cell_attr), state_params)

        # Add inventory items
        for item_id, item_attr in world_state.inventory.items():
            type_str = item_attr['item']
            if len(type_str)==0: # Empty inventory slot
                continue
            if type_str not in self.object_types:
                logger.info("Unknown inventory object type: %s" % type_str)
                type = PddlInventoryItemType()
            else:
                type = self.object_types[type_str]
            type.add_object_to_state(pddl_state, (item_id, item_attr), state_params)

        # Add other entities
        for entity, entity_attr in world_state.entities.items():
            type_str = entity_attr["type"]
            type = self.object_types[type_str]
            type.add_object_to_state(pddl_state, (entity, entity_attr), state_params)


        # A dictionary with global problem parameters
        state_params = dict()
        # state_params["has_platform"] = False
        # state_params["has_block"] = False
        # state_params["bird_index"] = 0
        # state_params["slingshot"] = slingshot
        # state_params["groundOffset"] = self.get_ground_offset(slingshot)
        # # Above line redundant since we're storing the slingshot also, but it seems easier to store it also to save computations of the offset everytime we use it.
        # state_params["pigs"] = set()
        # state_params["birds"] = set()
        # state_params["initial_state"] = False  # This marks that SBState describes the initial state. Used for setting the bird's location in the slingshot's location. TODO: Reconsider this design choice

        # # Add objects to problem
        # for obj in sb_state.objects.items():
        #     # Get type
        #     type_str = obj[1]['type']
        #     if 'bird' in type_str.lower() or (get_x_coordinate(obj) <= get_slingshot_x(slingshot) and not 'slingshot' in type_str):
        #         type = self.object_types["bird"]
        #     else:
        #         if type_str in self.object_types:
        #             type = self.object_types[type_str]
        #         else:
        #             logger.info("Unknown object type: %s" % type_str)
        #             # TODO Handle unknown objects in some way (Error? default object?)
        #             continue
        #     # print("(create pddl state) Object_type: " + str(type_str) + "/" + str(type) + " [" + str(get_x_coordinate(obj)) + ", " + str(get_y_coordinate(obj, state_params["groundOffset"])) + "] ")
        #
        #     # Add object of this type to the problem
        #     type.add_object_to_state(pddl_state, obj, state_params)

        return pddl_state

#
#     def create_pddl_action(self, state: PolycraftState, action: PolycraftAction):
#         ''' Create a PDDL+ action representing doing the given polycarft action in the given polycraft state '''
#
#         poly_action_to_pddl_action = dict()
#         poly_action_to_pddl_action[PolyTP : PddlTP]
#
#         return poly_action_to_pddl_action[action](action)
#
#
#
class PddlPolycraftAction():
    ''' A class representing a PDDL+ action in polycraft '''
    def to_pddl(self)->str:
        ''' This method should be implemented by sublcasses and output a string representation of the corresponding PDDL+ action '''
        raise NotImplementedError()

    def to_polycraft(self)->PddlPlusWorldChange:
        ''' This method should be implemented by sublcasses and output a string representation of the corresponding PDDL+ action '''
        raise NotImplementedError()

# class PddlTeleportToAndFace(PddlPolycraftAction):
#     def __init__(self, poly_action:PolyMoveToAndBreak):
#         self.poly_action = poly_action
#
#     def to_pddl(self)->PddlPlusWorldChange:
#         pddl_action = PddlPlusWorldChange(WorldChangeTypes.action)
#         pddl_action.name = "tp_to_and_fact"
#         pddl_action.parameters =  ["?c", "-", "cell"]
#         pddl_action.preconditions
#         pddl_action.effects
#         return '(tp_to_and_face ?c - cell)'
#
#     def to_polycarft(self, binding:dict)->PolycraftAction:
#         return TeleportAndFaceCell(cell)


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

#
# class PddlMoveToAndBreak(PddlPolycraftAction):
#     def __init__(self, poly_action:PolyMoveToAndBreak):
#         self.poly_action = poly_action
#
#     def to_pddl(self)->str:
#         return f'(move_and_break {self.poly_action.cell})'
#
# class PddlCollect(PddlPolycraftAction):
#     def __init__(self, poly_action:PolyCollect):
#         self.poly_action = poly_action
#
#     def to_pddl(self)->str:
#         return f'(collect)'
#
#
# class PddlTP(PddlPolycraftAction):
#     def __init__(self, poly_action:PolyTP):
#         self.cell = poly_action.cell
#         assert(poly_action.dist==1) # TODO: Design choice: allowing only distance 1 teleporting
#
#
#     def to_pddl(self): # Returns a pair of (lifted action, binding)
#         pddl_action = PddlPlusWorldChange(WorldChangeTypes.action)
#         pddl_action.name = "tp_to_cell"
#         pddl_action.parameters = ["?c", "-", "cell"]
#         return (pddl_action, {"?c" : self.cell})
# #
# # class PolyTP(PolycraftAction):
# #     """ Teleport to a position "dist" away from the xyz coordinates"""
# #
# #     def __init__(self, x: int, y: int, z: int, dist: int = 0):
# #         super().__init__()
# #         self.x = x
# #         self.y = y
# #         self.z = z
# #         self.dist = dist
# #
# #     def __str__(self):
# #         return "<PolyTP pos=({}, {}, {}) dist={} success={}>".format(self.x, self.y, self.z, self.dist,
# #                                                                      self.success)
# #
# #     def do(self, poly_client: poly.PolycraftInterface) -> dict:
# #         result = poly_client.TP_TO_POS(self.x, self.y, self.z, distance=self.dist)
# #         self.success = self.is_success(result)
# #         return result
#
#
