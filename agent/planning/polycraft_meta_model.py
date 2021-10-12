import copy
import math
import settings
import logging
import random
from agent.planning.meta_model import *
import worlds.polycraft_world as poly

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


class SteveType(PddlObjectType):
    ''' The Steve object. Based on the world state 'location' property '''
    def __init__(self):
        super(SteveType, self).__init__()
        self.pddl_type = "steve"

    def _compute_observable_obj_attributes(self, obj, problem_params:dict):
        obj_attributes = dict()
        obj_attributes["x"] = obj['pos'][0]
        obj_attributes["y"] = obj['pos'][1]
        obj_attributes["z"] = obj['pos'][2]
        obj_attributes["facing"] = obj["facing"]
        obj_attributes["yaw"] = obj["yaw"]
        obj_attributes["pitch"] = obj["pitch"]
        return obj_attributes

    def _get_name(self, obj):
        return 'steve'

class InventoryItemType(PddlObjectType):
    def __init__(self):
        super().__init__()
        self.pddl_type = "inventory_item"

    def _compute_observable_obj_attributes(self, obj, problem_params: dict):
        obj_attributes = dict()
        (item_id, item_attr) = obj
        for attr_name, attr_value in item_attr.items():
            if attr_name != "item":
                obj_attributes[attr_name]=attr_value

        if self.pddl_type != "inventory_item":
            obj_attributes["item_type"] = self.pddl_type

        return obj_attributes

    def _get_name(self, obj):
        (item_id, item_attr) = obj
        return "inventory_{}".format(item_id, item_attr["item"])

class IronPickaxeType(InventoryItemType):
    def __init__(self):
        super().__init__()
        self.pddl_type = "iron_pickaxe"

class GameMapCellType(PddlObjectType):
    def __init__(self,block_life_multiplier = 1.0, block_mass_coeff=1.0):
        super().__init__()
        self.pddl_type = "cell"

    def _compute_observable_obj_attributes(self, obj, problem_params:dict):
        obj_attributes = dict()
        (cell_id, cell_attr)=obj

        for attr_name, attr_value in cell_attr.items():
            if attr_name != 'name':
                obj_attributes[attr_name] = attr_value

        x, y, z = cell_id.split(",")
        obj_attributes["x"] = int(x)
        obj_attributes["y"] = int(y)
        obj_attributes["z"] = int(z)

        if self.pddl_type != "cell":
            obj_attributes["cell_type"] = self.pddl_type

        return obj_attributes

    def _get_name(self, obj):
        (cell_id, cell_attr) = obj
        return "cell_{}".format("_".join(cell_id.split(",")))


class BedrockType(GameMapCellType):
    def __init__(self):
        super().__init__()
        self.pddl_type = "bedrock"

class AirType(GameMapCellType):
    def __init__(self):
        super().__init__()
        self.pddl_type = "air"

    def _compute_observable_obj_attributes(self, obj, problem_params:dict):
        return dict() # Ignoring air blocks in the PDDL

class LogType(GameMapCellType):
    def __init__(self):
        super().__init__()
        self.pddl_type = "log"

class DiamondOreType(GameMapCellType):
    def __init__(self):
        super().__init__()
        self.pddl_type = "diamond_ore"

class PlasticChestType(GameMapCellType):
    def __init__(self):
        super().__init__()
        self.pddl_type = "plastic_chest"

class CraftingTableType(GameMapCellType):
    def __init__(self):
        super().__init__()
        self.pddl_type = "crafting_table"

class BlockOfPlatinumType(GameMapCellType):
    def __init__(self):
        super().__init__()
        self.pddl_type = "block_of_platinum"

class WoodenDoorType(GameMapCellType):
    def __init__(self):
        super().__init__()
        self.pddl_type = "wooden_door"


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
        self.object_types = dict()
        self.object_types["minecraft:bedrock"]=BedrockType()
        self.object_types["minecraft:air"]=AirType()
        self.object_types["minecraft:log"]=LogType()
        self.object_types["minecraft:diamond_ore"]=DiamondOreType()
        self.object_types["polycraft:plastic_chest"]=PlasticChestType()
        self.object_types["minecraft:iron_pickaxe"]=IronPickaxeType()
        self.object_types["minecraft:crafting_table"]=CraftingTableType()

        self.object_types["minecraft:wooden_door"] = WoodenDoorType()
        self.object_types["polycraft:block_of_platinum"] = BlockOfPlatinumType()


    def create_pddl_domain(self, observed_state) -> PddlPlusDomain:
        ''' Create a PDDL+ domain for the given observed state '''
        domain_file = "{}/{}".format(str(self.docker_path), "polycraft_domain_template.pddl")
        domain_parser = PddlDomainParser()
        pddl_domain = domain_parser.parse_pddl_domain(domain_file)

        # Add actions for recipes

        # Add actions for trades


    def create_pddl_problem(self, world_state : poly.PolycraftState):
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

        # Add objects


        # A dictionary with current state parameters that are not object related
        state_params = dict()

        # Add Steve location to init state
        steve_type = SteveType()
        steve_type.add_object_to_problem(pddl_problem, world_state.location, state_params)

        # Add game map cells
        for cell, cell_attr in world_state.game_map.items():
            type_str = cell_attr['name']

            if type_str not in self.object_types:
                logger.info("Unknown game map cell type: %s" % type_str)
                type = GameMapCellType()
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
                type = InventoryItemType()
            else:
                type = self.object_types[type_str]
            type.add_object_to_problem(pddl_problem, (item_id, item_attr), state_params)

        # Currently modeling other entities only via their trades TODO: Re-consider this

        # Add goal
        pddl_problem.goal.append(['pogo_created',])

        return pddl_problem

    def create_pddl_state(self, world_state: poly.PolycraftState) -> PddlPlusState:
        ''' Translate the given observed world state to a PddlPlusState object '''

        pddl_state = PddlPlusState()

        # A dictionary with current state parameters that are not object related
        state_params = dict()

        # Add Steve location to state
        steve_type = SteveType()
        steve_type.add_object_to_state(pddl_state, world_state.location, state_params)

        # Add game map cells
        for cell, cell_attr in world_state.game_map.items():
            type_str = cell_attr['name']

            if type_str not in self.object_types:
                logger.info("Unknown game map cell type: %s" % type_str)
                type = GameMapCellType()
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
                type = InventoryItemType()
            else:
                type = self.object_types[type_str]
            type.add_object_to_state(pddl_state, (item_id, item_attr), state_params)

        # Currently modeling other entities only via their trades TODO: Re-consider this


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
