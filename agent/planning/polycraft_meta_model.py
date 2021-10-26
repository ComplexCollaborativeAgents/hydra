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


###### PDDL ACTIONS, PROCESSES, AND EVENTS
class PddlPolycraftAction():
    ''' A class representing a PDDL+ action in polycraft '''
    def to_pddl(self, meta_model: MetaModel)->PddlPlusWorldChange:
        ''' This method should be implemented by sublcasses and output a string representation of the corresponding PDDL+ action '''
        raise NotImplementedError()

    def to_polycraft(self, parameter_binding:dict)->PolycraftAction:
        ''' This method should be implemented by sublcasses and output a string representation of the corresponding PDDL+ action '''
        raise NotImplementedError()

class PddlPlaceTreeTapAction(PddlPolycraftAction):
    ''' An action corresponding to placing a tree tap
        (:action place_tree_tap
            :parameters (?at - cell ?near_to - cell)
            :precondition (and
                (isAccessible ?at)
                (isAccessible ?near_to)
                (adjacent ?at ?near_to)
                (cell_type ?at {BlockType.AIR.value})
                (cell_type ?near_to {BlockType.LOG.value})
                (>= (count_{ItemType.TREE_TAP.name}) 1)
            )
            :effect (and
                (decrease (count_{ItemType.TREE_TAP}) 1)
                (assign (cell_type ?to) {ItemType.TREE_TAP.value})
            )
        )
    '''
    def to_pddl(self, meta_model: MetaModel)->PddlPlusWorldChange:
        pddl_action = PddlPlusWorldChange(WorldChangeTypes.action)
        pddl_action.name = "place_tree_tap"
        pddl_action.parameters.append(["?at", "-", "cell", "?near_to", "-", "cell"])
        pddl_action.preconditions.append(["isAccessible", "?at"])
        pddl_action.preconditions.append(["isAccessible", "?near_to"])
        pddl_action.preconditions.append(["adjacent", "?at", "?near_to"])

        air_idx = meta_model.block_type_to_idx[BlockType.AIR.value]
        pddl_action.preconditions.append(["=", ["cell_type", "?at"], f"{air_idx}"])
        log_idx = meta_model.block_type_to_idx[BlockType.LOG.value]
        pddl_action.preconditions.append(["=", ["cell_type", "?near_to"], f"{log_idx}"])
        pddl_action.preconditions.append([">=", [f"count_{ItemType.TREE_TAP.value}",], "1"])

        pddl_action.effects.append(["decrease", [f"count_{ItemType.TREE_TAP.value}",], "1"])
        tree_tap_idx = meta_model.block_type_to_idx[BlockType.TREE_TAP.value]
        pddl_action.effects.append(["assign", ["cell_type", "?to"], f"{tree_tap_idx}"])
        return pddl_action

    def to_polycraft(self, parameter_binding:dict)->PolycraftAction:
        return PlaceTreeTap(parameter_binding["at"])

class PddlCollectAction(PddlPolycraftAction):
    ''' An action corresponding to collecting an item from a cell (COLLECT in the Polycraft API)
    (:action collect_{item_type}_from_{block_type}
        :parameters (?c - cell)
        :precondition( and
            (isAccessible ?c)
            (cell_type ?c {block_type})
        )
        :effect( and
            (increase(count_{item_type}) 1)
        )
    )
    '''
    def __init__(self, collect_from_block_of_type: str, item_type_to_collect : str, quantity):
        self.item_type_to_collect = item_type_to_collect
        self.collect_from_block_type = collect_from_block_of_type
        self.quantity = quantity

    def to_pddl(self, meta_model: MetaModel)->PddlPlusWorldChange:
        pddl_action = PddlPlusWorldChange(WorldChangeTypes.action)
        pddl_action.name = f"collect_{self.item_type_to_collect}_from_{self.collect_from_block_type}"
        pddl_action.parameters.append(["?c", "-", "cell"])
        pddl_action.preconditions.append(["isAccessible", "?c"])

        cell_type_idx = meta_model.block_type_to_idx[self.collect_from_block_type]
        pddl_action.preconditions.append(["=", ["cell_type", "?c"], f"{cell_type_idx}"])
        pddl_action.effects.append(["increase", [f"count_{self.item_type_to_collect}", ], str(self.quantity)])
        return pddl_action

    def to_polycraft(self, parameter_binding: dict) -> PolycraftAction:
        return TeleportToAndCollect(parameter_binding['?c'])

class PddlBreakAction(PddlPolycraftAction):
    ''' An action for moving to a cell, breaking it, and collecting the resulting item.

    ; TELEPORT_TO AND BREAK
    (:action break_{block_type}
        :parameters (?c - cell)
        :precondition (and
            (isAccessible ?c)
            (cell_type ?c {block_type})
            (= (selectedItem) {ItemType.IRON_PICKAXE.value})
        )
        :effect (and
            (increase (count_{item_type}) {items_per_block})
            (cell_type ?c {BlockType.AIR.value})
        )
    )
    '''
    def __init__(self, block_type_to_break : str, item_type_to_collect: str, items_per_block: int, needs_iron_pickaxe=False):
        self.block_type_to_break = block_type_to_break
        self.item_type_to_collect = item_type_to_collect
        self.items_per_block = items_per_block
        self.needs_iron_pickaxe = needs_iron_pickaxe

    def to_pddl(self, meta_model: MetaModel)->PddlPlusWorldChange:
        pddl_action = PddlPlusWorldChange(WorldChangeTypes.action)
        pddl_action.name = f"break_{self.block_type_to_break}"
        pddl_action.parameters.append(["?c", "-", "cell"])
        pddl_action.preconditions.append(["isAccessible", "?c"])

        cell_type_idx = meta_model.block_type_to_idx[self.block_type_to_break]
        pddl_action.preconditions.append(["=", ["cell_type", "?c"], f"{cell_type_idx}"])
        if self.needs_iron_pickaxe:
            iron_pickaxe_idx = meta_model.item_type_to_idx[ItemType.IRON_PICKAXE.value]
            pddl_action.preconditions.append(["=", ["selectedItem",], f"{iron_pickaxe_idx}"])
        pddl_action.effects.append(["increase", [f"count_{self.item_type_to_collect}", ], str(self.items_per_block)])
        return pddl_action

    def to_polycraft(self, parameter_binding: dict) -> PolycraftAction:
        return TeleportToBreakAndCollect(parameter_binding['?c'])

class PddlSelectAction(PddlPolycraftAction):
    ''' An action for selecting an item from the inventory
        ; SELECT
        (:action select_{item_type.name}
            :precondition (and
                (>= (count_{item_type.name}) 1)
            )
            :effect (and
                (assign (selectedItem) {item_type.value})
            )
        )
    '''
    def __init__(self, item_type: str):
        self.item_type_name = item_type

    def to_pddl(self, meta_model: MetaModel)->PddlPlusWorldChange:
        pddl_action = PddlPlusWorldChange(WorldChangeTypes.action)
        pddl_action.name = f"select_{self.item_type_name}"
        pddl_action.preconditions.append([">=", [f"count_{self.item_type_name}",], "1"])

        item_type_idx = meta_model.item_type_to_idx[self.item_type_name]
        pddl_action.effects.append(
            ["assign", ["selectedItem", ], f"{item_type_idx}"])
        return pddl_action

    def to_polycraft(self, parameter_binding: dict) -> PolycraftAction:
        return PolySelectItem(self.item_type_name)

class PddlCraftAction(PddlPolycraftAction):
    ''' An action for crafting an item
        ; CRAFT 1
        (:action craft_recipe_{recipe_idx}
            :parameters (?from - cell)
            :precondition( and
                (cell_type ?from {BlockType.CRAFTING_TABLE.value})
                (isAccessible ?from)
                (>= (count_{recipe_input})
                {input_quantity})
            )
            :effect( and
                (increase(count_{recipe_output})
                {output_quantity})
                (decrease(count_{recipe_input})
                {input_quantity})
            )
        )
    '''
    def __init__(self, recipe_idx: int, recipe, needs_crafting_table:bool = False):
        self.recipe_idx = recipe_idx
        self.recipe = recipe
        self.needs_crafting_table = needs_crafting_table

    def to_pddl(self, meta_model: MetaModel)->PddlPlusWorldChange:
        pddl_action = PddlPlusWorldChange(WorldChangeTypes.action)
        pddl_action.name = f"craft_recipe_{self.recipe_idx}"

        if self.needs_crafting_table: # TODO: Not robust to problems with multiple crafting tables
            pddl_action.parameters.append(["?from", "-", "cell"])
            crafting_table_idx = meta_model.block_type_to_idx[BlockType.CRAFTING_TABLE.value]
            pddl_action.preconditions.append(["=", ["cell_type", "?from",], f"{crafting_table_idx}"])
            pddl_action.preconditions.append(["isAccessible", "?from"])

        for item_type, quantity in get_ingredients_for_recipe(self.recipe).items():
            pddl_action.preconditions.append([">=", [f"count_{item_type}"], str(quantity)])
            pddl_action.effects.append(["decrease", [f"count_{item_type}"], str(quantity)])

        for item_type, quantity in get_outputs_of_recipe(self.recipe).items():
            pddl_action.effects.append(["increase", [f"count_{item_type}"], str(quantity)])

        return pddl_action

    def to_polycraft(self, parameter_binding: dict) -> PolycraftAction:
        if self.needs_crafting_table:
            return TeleportToTableAndCraft(parameter_binding["?from"], self.recipe)
        else:
            return PolyCraftItem.create_action(self.recipe)

class PddlTradeAction(PddlPolycraftAction):
    ''' An action for trading an item with a trader
        ; TRADE
        (:action trade_recipe_{trader_id}_{trade_idx}
            :parameters (?trader_loc - cell)
            :precondition (and
                (trader_{trader_id}_at ?trader_loc)
                (isAccessible ?trader_loc)
                (>= (count_{trade_input}) {input_quantity})
            )
            :effect (and
                (increase (count_{trade_output}) {output_quantity})
                (decrease (count_{trade_input}) {input_quantity})
            )
        )
    '''

    def __init__(self, trader_id:str, trade_idx: int, trade):
        self.trader_id = trader_id
        self.trade_idx = trade_idx
        self.trade = trade

    def to_pddl(self, meta_model: MetaModel)->PddlPlusWorldChange:
        pddl_action = PddlPlusWorldChange(WorldChangeTypes.action)
        pddl_action.name = f"trade_recipe_{self.trader_id}_{self.trade_idx}"
        pddl_action.parameters.append(["?trader_loc","-", "cell"])
        pddl_action.preconditions.append(["isAccessible", "?trader_loc"])
        pddl_action.preconditions.append([f"trader_{self.trader_id}_at", "?trader_loc"])

        for input in self.trade["inputs"]:
            item_type = input['Item']
            quantity = input['stackSize']
            pddl_action.preconditions.append([">=", [f"count_{item_type}"], str(quantity)])
            pddl_action.effects.append(["decrease", [f"count_{item_type}"], str(quantity)])

        for output in self.trade["outputs"]:
            item_type = output['Item']
            quantity = output['stackSize']
            pddl_action.effects.append(["increase", [f"count_{item_type}"], str(quantity)])

        return pddl_action

    def to_polycraft(self, parameter_binding: dict) -> PolycraftAction:
        return TeleportToTraderAndTrade(self.trader_id, self.trade)

class PddlPolycraftEvent():
    ''' A class representing a PDDL+ action in polycraft '''

    def to_pddl(self) -> PddlPlusWorldChange:
        ''' This method should be implemented by sublcasses and output a string representation of the corresponding PDDL+ action '''
        raise NotImplementedError()

class CellAccessibleEvent(PddlPolycraftEvent):
    def to_pddl(self, meta_model: MetaModel):
        pddl_event = PddlPlusWorldChange(WorldChangeTypes.event)
        pddl_event.name = "cell_accessible"
        pddl_event.parameters.append(["?c1","-", "cell", "?c2", "-", "cell"])
        pddl_event.preconditions.append(["isAccessible", "?c1"])
        pddl_event.preconditions.append(["not", ["isAccessible", "?c2"]])
        air_idx = meta_model.block_type_to_idx[BlockType.AIR.value]
        pddl_event.preconditions.append(["=", ["cell_type", "?c1"], f"{air_idx}"])
        pddl_event.preconditions.append(["adjacent", "?c1", "?c2"])
        pddl_event.effects.append(["isAccessible", "?c2"])
        return pddl_event

###### PDDL OBJECTS AND FLUENTS

##### Classes and constructs to help build meta models
class PolycraftObjectType():
    ''' A generator for Pddl Objects. Accepts an object from the domain and adds the corresponding objects and fluents to the PDDL problem. '''

    def __init__(self):
        self.hyper_parameters = dict()
        self.pddl_type="object" # This the PDDL+ type of this object.

    def get_object_name(self, obj):
        ''' Return the object name in PDDL '''
        raise NotImplementedError("Subclass should implement this: generate a pddl name for the given world object")

    def add_object_to_problem(self, prob: PddlPlusProblem, obj, problem_params:dict):
        ''' Populate a PDDL+ problem with details about this object '''
        obj_name = self.get_object_name(obj)
        if self.pddl_type is not None: # None type means this object should not be added to the problem objects list.
            prob.objects.append([obj_name, self.pddl_type])
        fluent_to_value = self._compute_obj_fluents(obj, problem_params)
        for fluent_name, fluent_value in fluent_to_value.items():
            fluent_name_as_list = list(fluent_name)
            # If attribute is Boolean no need for an "=" sign
            if isinstance(fluent_value,  bool):
                if fluent_value==True:

                    prob.init.append(fluent_name_as_list)
                else: # value == False
                    prob.init.append(['not', fluent_name_as_list])
            else: # Attribute is a number
                prob.init.append(['=', fluent_name_as_list, fluent_value])

    def add_object_to_state(self, pddl_state: PddlPlusState, obj, state_params:dict):
        ''' Populate a PDDL+ state with details about this object '''
        fluent_to_value = self._compute_obj_fluents(obj, state_params)
        for fluent_name, fluent_value in fluent_to_value.items():
            # If attribute is Boolean no need for an "=" sign
            if isinstance(fluent_value, bool):
                if fluent_value == True:
                    pddl_state.boolean_fluents.add(fluent_name)
                # TODO: Think how to handle booean fluents with False value. Not as trivial as it sounds
            else:  # Attribute is a number
                pddl_state.numeric_fluents[fluent_name] = fluent_value

    def _compute_obj_fluents(self, obj, params:dict)->dict:
        ''' Maps fluent_name to fluent_value for all fluents created for this object'''
        raise NotImplementedError("Subclass should implement. Return a dict mapping fluent name to value")

class PddlGameMapCellType(PolycraftObjectType):
    def __init__(self, type_idx=-1, relevant_attributes:list = ["isAccessible"]):
        super().__init__()
        self.pddl_type = "cell"
        self.type_idx = type_idx
        self.relevant_attributes = relevant_attributes

    @staticmethod
    def get_cell_object_name(cell_id:str):
        ''' Return the object name in PDDL for the given cell '''
        return "cell_{}".format("_".join(cell_id.split(",")))

    def get_object_name(self, obj):
        ''' Return the object name in PDDL '''
        (cell_id, cell_attr) = obj
        return PddlGameMapCellType.get_cell_object_name(cell_id)

    def _compute_obj_fluents(self, obj, params:dict)->dict:
        ''' Maps fluent_name to fluent_value for all fluents created for this object'''
        cell_name = self.get_object_name(obj)
        fluent_to_value = dict()
        fluent_to_value[("cell_type", cell_name)] = self.type_idx

        (cell_id, cell_attr) = obj
        for attribute, attribute_value in cell_attr.items():
            # If attribute is Boolean no need for an "=" sign
            if attribute in self.relevant_attributes:
                fluent_to_value[(attribute, cell_name)]=attribute_value

        world_state = params["world_state"]
        for adjacent_cell in get_adjacent_cells(cell_id):
            if adjacent_cell in world_state.game_map:
                fluent_to_value[("adjacent", cell_name, PddlGameMapCellType.get_cell_object_name(adjacent_cell))]=True

        return fluent_to_value





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

        # Maps a cell type to what we get if we break it. The latter is in the form of a pair (item type, quantity).
        self.break_block_to_outcome = dict()
        self.break_block_to_outcome[BlockType.LOG.value] = (ItemType.LOG.value, 1)
        self.break_block_to_outcome[BlockType.BLOCK_OF_PLATINUM.value] = (ItemType.BLOCK_OF_PLATINUM.value, 1)
        self.break_block_to_outcome[BlockType.DIAMOND_ORE.value] = (ItemType.DIAMOND.value, 1)

        # Maps a cell type to what we get if we collect from it. The latter is in the form of a pair (item type, quantity).
        self.collect_block_to_outcome = dict()
        self.collect_block_to_outcome[BlockType.TREE_TAP.value] = (ItemType.SACK_POLYISOPRENE_PELLETS.value, 1)

        # List of cell types that require an iron pickaxe to break
        self.needs_iron_pickaxe = list()
        self.needs_iron_pickaxe.append(BlockType.BLOCK_OF_PLATINUM.value)
        self.needs_iron_pickaxe.append(BlockType.DIAMOND_ORE.value)

        # List of items that one may select
        self.selectable_items = list()
        self.selectable_items.append(ItemType.IRON_PICKAXE.value)

        # Assign indices to block types and item types
        self.block_type_to_idx = dict()
        type_idx = 0
        for block_type in BlockType:
            self.block_type_to_idx[block_type.value]=type_idx
            type_idx = type_idx + 1

        self.item_type_to_idx = dict()
        type_idx = 0
        for item_type in ItemType:
            self.item_type_to_idx[item_type.value]=type_idx
            type_idx = type_idx + 1

    def create_pddl_domain(self, world_state:PolycraftState) -> PddlPlusDomain:
        ''' Create a PDDL+ domain for the given observed state '''
        # domain_file = "{}/{}".format(str(self.docker_path), "polycraft_domain_template.pddl")
        # domain_parser = PddlDomainParser()
        # pddl_domain = PddlPlusDomain()

        pddl_domain = PddlPlusDomain()
        pddl_domain.name = "polycraft"
        pddl_domain.requirements = [":typing", ":disjunctive-preconditions", ":fluents", ":time", ":negative-preconditions"]
        pddl_domain.types.append("cell")

        # Add predicates
        pddl_domain.predicates.append(["isAccessible", "?c", "-", "cell"])
        pddl_domain.predicates.append(["adjacent", "?c1", "-", "cell", "?c2", "-", "cell"])
        for trader_id in set(world_state.trades.keys()):
            pddl_domain.predicates.append([f"trader_{trader_id}_at", "?c", "-", "cell"])

        # Add functions
        pddl_domain.functions.append(["cell_type", "?c","-","cell"])
        pddl_domain.functions.append(["selectedItem",])
        for item in self.item_type_to_idx:
            pddl_domain.functions.append([f"count_{item}",])

        # Add actions
        pddl_actions = []
        pddl_actions.append(PddlPlaceTreeTapAction())

        for item_type in self.selectable_items:
            pddl_actions.append(PddlSelectAction(item_type=item_type))

        # Add blocks to break and get items
        for block_type, outcome in self.break_block_to_outcome.items():
            item_type = outcome[0]
            quantity = outcome[1]
            if block_type in self.needs_iron_pickaxe:
                needs_iron_pickaxe=True
            else:
                needs_iron_pickaxe = False
            pddl_actions.append(PddlBreakAction(block_type, item_type, quantity,
                                                needs_iron_pickaxe=needs_iron_pickaxe))

        # Add collect item to outcome
        for block_type, outcome in self.collect_block_to_outcome.items():
            item_type = outcome[0]
            quantity = outcome[1]
            pddl_actions.append(PddlCollectAction(block_type, item_type, quantity))

        # Add recipes
        for recipe_idx, recipe in enumerate(world_state.recipes):
            craft_action = PolyCraftItem.create_action(recipe)
            if len(craft_action.recipe)==9: # Need a crafting table
                pddl_actions.append(PddlCraftAction(recipe_idx, recipe,needs_crafting_table=True))
            else:
                pddl_actions.append(PddlCraftAction(recipe_idx, recipe, needs_crafting_table=False))

        # Add trades
        for trader_id, trades in world_state.trades.items():
            for trade_idx, trade in enumerate(trades):
                pddl_actions.append(PddlTradeAction(trader_id, trade_idx, trade))

        # Add all actions to the domain
        for pddl_action in pddl_actions:
            pddl_domain.actions.append(pddl_action.to_pddl(self))

        pddl_domain.events.append(CellAccessibleEvent().to_pddl(self))

        return pddl_domain

    def _should_ignore_cell(self, cell:str, world_state:PolycraftState):
        ''' An optimization step: identify cells that are not needed for the problem solving '''
        cell_types_to_ignore = [BlockType.BEDROCK.value]
        type_str = world_state.game_map[cell]['name']
        if type_str in cell_types_to_ignore:
            return True

        # Check if all neighbors are air
        all_neighbors_air = True
        if type_str == BlockType.AIR.value:
            for neighbor_cell in get_adjacent_cells(cell):
                if neighbor_cell in world_state.game_map:
                    neighbor_type = world_state.game_map[neighbor_cell]['name']
                    if neighbor_type!=BlockType.AIR.value:
                        all_neighbors_air=False
                        break
        if all_neighbors_air:
            return True

        return False

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
        problem_params["world_state"]= world_state

        # Add game map cells
        cell_types_to_ignore = [BlockType.BEDROCK.value]
        for cell, cell_attr in world_state.game_map.items():
            # Pruning cells to gain efficiency
            if self._should_ignore_cell(cell, world_state):
                continue

            # Add cell to problem
            type_str = cell_attr['name']
            if type_str not in self.block_type_to_idx:
                logger.info("Unknown game map cell type: %s" % type_str)
                type = PddlGameMapCellType(type_idx=-1)
            else:
                type = PddlGameMapCellType(type_idx=self.block_type_to_idx[type_str])
            type.add_object_to_problem(pddl_problem, (cell, cell_attr), problem_params)

        # Add inventory items
        for item_type in self.item_type_to_idx.keys():
            count = world_state.count_items_of_type(item_type)
            pddl_problem.init.append(['=', [f"count_{item_type}",], f"{count}"])

        # Add selected item
        select_item = world_state.get_selected_item()
        fluent_name = "selectedItem"
        selected_item_idx = -1
        if select_item is not None and select_item in self.item_type_to_idx:
            selected_item_idx = self.item_type_to_idx[select_item]
            pddl_problem.init.append(["=", [fluent_name, ], str(selected_item_idx)])
        else:
            pddl_problem.init.append(["=", [fluent_name, ], str(-1)])

        # Add other entities
        for entity, entity_attr in world_state.entities.items():
            type_str = entity_attr["type"]
            if EntityType.TRADER.value == type_str:
                entity_cell = coordinates_to_cell(entity_attr["pos"])
                cell_name = PddlGameMapCellType.get_cell_object_name(entity_cell)
                pddl_problem.init.append([f"trader_{entity}_at", cell_name])

        # Add goal and metric
        pddl_problem.goal.append(['>', [f"count_{ItemType.WOODEN_POGO_STICK.value}",], "0"])
        pddl_problem.metric = "minimize(total-time)"

        return pddl_problem

    def create_pddl_state(self, world_state: PolycraftState) -> PddlPlusState:
        ''' Translate the given observed world state to a PddlPlusState object '''
        raise NotImplementedError("todo")