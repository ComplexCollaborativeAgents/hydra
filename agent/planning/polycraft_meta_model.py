import logging

from agent.planning.meta_model import *
from agent.planning.meta_model import MetaModel
from agent.planning.pddl_plus import PddlPlusWorldChange
from agent.planning.polycraft_planning.actions import *
from worlds.polycraft_world import *
from agent.planning.pddl_plus import *
from worlds.polycraft_world import PolycraftAction

logging.basicConfig(format='%(name)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PolycraftPDDL")
logger.setLevel(logging.DEBUG)


# CONSTANTS
class PddlType(enum.Enum):
    cell = "cell"
    door_cell = "door_cell"
    safe_cell = "safe_cell"


class Predicate(enum.Enum):
    """ Note: the first prameter in the list is needed: otherwise python will merge enum elements. """
    isAccessible = ["isAccessible", ("?c", PddlType.cell.name)]
    # adjacent = ["adjacent", ("?c1", PddlType.cell.name), ("?c2", PddlType.cell.name)]

    door_is_accessible = ["door_is_accessible", ("?c", PddlType.door_cell.name)]
    # adjacent_to_door = ["adjacent_to_door", ("?c1", PddlType.cell.name), ("?c2", PddlType.door_cell.name)]
    open = ["open", ("?c", PddlType.door_cell.name)]
    passed_door = ["passed_door", ("?c", PddlType.door_cell.name)]

    safe_is_accessible = ["safe_is_accessible", ("?c", PddlType.safe_cell.name)]
    # adjacent_to_safe = ["adjacent_to_safe", ("?c1", PddlType.cell.name), ("?c2", PddlType.safe_cell.name)]
    safe_collected = ["safe_collected", ("?c", PddlType.safe_cell.name)]
    safe_open = ["safe_open", ("?c", PddlType.safe_cell.name)]

    def to_pddl(self) -> list:
        """ Returns this predicate in a list format as expected by the pddl domain object """
        predicate_as_list = [self.name]
        for (param_name, param_type) in self.value[1:]:
            predicate_as_list.extend([param_name, "-", param_type])
        return predicate_as_list


class Function(enum.Enum):
    """ Note: the first prameter in the list is needed: otherwise python will merge enum elements. """
    cell_type = ["cell_type", ("?c", PddlType.cell.name)]
    cell_x = ["cell_x", ("?c", PddlType.cell.name)]
    cell_z = ["cell_z", ("?c", PddlType.cell.name)]
    door_cell_type = ["door_cell_type", ("?c", PddlType.door_cell.name)]
    selectedItem = ["selectedItem"]
    Steve_x = ["steve_x"]
    Steve_z = ["steve_z"]

    def to_pddl(self) -> list:
        """ Returns this function in a list format as expected by the pddl domain object """
        function_as_list = [self.name]
        for (param_name, param_type) in self.value[1:]:
            function_as_list.extend([param_name, "-", param_type])
        return function_as_list


###### PDDL OBJECTS AND FLUENTS

##### Classes and constructs to help build meta models
class PolycraftObjectType:  #TODO each meta-model file has it's own definition of this class - consolidate
    """ A generator for Pddl Objects. Accepts an object from the domain and adds the corresponding objects and fluents to the PDDL problem. """

    def __init__(self):
        self.hyper_parameters = dict()
        self.pddl_type = "object"  # This the PDDL+ type of this object.

    def get_object_name(self, obj):
        """ Return the object name in PDDL """
        raise NotImplementedError("Subclass should implement this: generate a pddl name for the given world object")

    def add_object_to_pddl(self, obj, prob: PddlPlusProblem, params: dict):
        """ Populate a PDDL+ problem with details about this object """
        obj_name = self.get_object_name(obj)
        if self.pddl_type is not None:  # None type means this object should not be added to the problem objects list.
            prob.objects.append([obj_name, self.pddl_type])
        fluent_to_value = self._compute_obj_fluents(obj, params)
        for fluent_name, fluent_value in fluent_to_value.items():
            fluent_name_as_list = list(fluent_name)
            # If attribute is Boolean no need for an "=" sign
            if isinstance(fluent_value, bool):
                if fluent_value == True:

                    prob.init.append(fluent_name_as_list)
                else:  # value == False
                    prob.init.append(['not', fluent_name_as_list])
            else:  # Attribute is a number
                prob.init.append(['=', fluent_name_as_list, fluent_value])

    def add_object_to_state(self, pddl_state: PddlPlusState, obj, state_params: dict):
        """ Populate a PDDL+ state with details about this object """
        fluent_to_value = self._compute_obj_fluents(obj, state_params)
        for fluent_name, fluent_value in fluent_to_value.items():
            # If attribute is Boolean no need for an "=" sign
            if isinstance(fluent_value, bool):
                if fluent_value == True:
                    pddl_state.boolean_fluents.add(fluent_name)
                # TODO: Think how to handle booean fluents with False value. Not as trivial as it sounds
            else:  # Attribute is a number
                pddl_state.numeric_fluents[fluent_name] = fluent_value

    def _compute_obj_fluents(self, obj, params: dict) -> dict:
        """ Maps fluent_name to fluent_value for all fluents created for this object"""
        raise NotImplementedError("Subclass should implement. Return a dict mapping fluent name to value")


class PddlGameMapCellType(PolycraftObjectType):
    def __init__(self, type_idx=-1, relevant_attributes=None):
        super().__init__()
        if relevant_attributes is None:
            relevant_attributes = [Predicate.isAccessible.name, Function.cell_x.name, Function.cell_z.name]
        self.pddl_type = "cell"
        self.type_idx = type_idx
        self.relevant_attributes = relevant_attributes

    @staticmethod
    def get_cell_object_name(cell_id: str):
        """ Return the object name in PDDL for the given cell """
        return "cell_{}".format("_".join(cell_id.split(",")))

    def get_object_name(self, obj):
        """ Return the object name in PDDL """
        (cell_id, cell_attr) = obj
        return PddlGameMapCellType.get_cell_object_name(cell_id)

    def _compute_obj_fluents(self, obj, params: dict) -> dict:
        """ Maps fluent_name to fluent_value for all fluents created for this object"""
        cell_name = self.get_object_name(obj)
        fluent_to_value = dict()
        fluent_to_value[(Function.cell_type.name, cell_name)] = self.type_idx

        (cell_id, cell_attr) = obj
        for attribute, attribute_value in cell_attr.items():
            # If attribute is Boolean no need for an "=" sign
            if attribute in self.relevant_attributes:
                fluent_to_value[(attribute, cell_name)] = attribute_value

        # # Cell adjacency info
        # world_state = params["world_state"]
        # active_cells = params["active_cells"]
        # known_cells = world_state.get_known_cells()
        # for adjacent_cell in get_adjacent_cells(cell_id):
        #     if adjacent_cell in active_cells:
        #         adjacent_cell_type = known_cells[adjacent_cell]["name"]
        #         if adjacent_cell_type in [BlockType.BEDROCK.value, BlockType.WOODER_DOOR.value, BlockType.SAFE.value]:
        #             continue
        #         adjacent_cell_name = PddlGameMapCellType.get_cell_object_name(adjacent_cell)
        #         fluent_to_value[(Predicate.adjacent.name, cell_name, adjacent_cell_name)] = True
        return fluent_to_value


class PddlDoorCellType(PddlGameMapCellType):
    def __init__(self, type_idx=-1):
        super().__init__(type_idx, [])
        self.pddl_type = PddlType.door_cell.name

    def _compute_obj_fluents(self, obj, params: dict) -> dict:
        """ Maps fluent_name to fluent_value for all fluents created for this object"""
        cell_name = self.get_object_name(obj)
        fluent_to_value = dict()
        fluent_to_value[(Function.door_cell_type.name, cell_name)] = self.type_idx

        (cell_id, cell_attr) = obj
        for attribute, attribute_value in cell_attr.items():
            # If attribute is Boolean no need for an "=" sign
            if attribute == Predicate.isAccessible.name:
                fluent_to_value[(Predicate.door_is_accessible.name, cell_name)] = attribute_value
            elif attribute == Predicate.open.name:
                if attribute_value.upper() == "TRUE":
                    fluent_to_value[(Predicate.door_is_accessible.name, cell_name)] = True

        # Handle adjacency
        # world_state = params["world_state"]
        # active_cells = params["active_cells"]
        # known_cells = world_state.get_known_cells()
        # for adjacent_cell in get_adjacent_cells(cell_id):
        #     if adjacent_cell in active_cells:
        #         adjacent_cell_type = known_cells[adjacent_cell]["name"]
        #         if adjacent_cell_type in [BlockType.BEDROCK.value, BlockType.WOODER_DOOR.value]:
        #             continue
        #         adjacent_cell_name = PddlGameMapCellType.get_cell_object_name(adjacent_cell)
        #         fluent_to_value[(Predicate.adjacent_to_door.name, adjacent_cell_name, cell_name)] = True

        return fluent_to_value


class PddlSafeCellType(PddlGameMapCellType):
    def __init__(self, type_idx=-1):
        super().__init__(type_idx, [])
        self.pddl_type = PddlType.safe_cell.name

    def _compute_obj_fluents(self, obj, params: dict) -> dict:
        """ Maps fluent_name to fluent_value for all fluents created for this object"""
        cell_name = self.get_object_name(obj)
        fluent_to_value = dict()

        (cell_id, cell_attr) = obj
        for attribute, attribute_value in cell_attr.items():
            # If attribute is Boolean no need for an "=" sign
            if attribute == Predicate.isAccessible.name:
                fluent_to_value[(Predicate.safe_is_accessible.name, cell_name)] = attribute_value

        # # Handle adjacency
        # world_state = params["world_state"]
        # active_cells = params["active_cells"]
        # known_cells = world_state.get_known_cells()
        # for adjacent_cell in get_adjacent_cells(cell_id):
        #     if adjacent_cell in active_cells:
        #         adjacent_cell_type = known_cells[adjacent_cell]["name"]
        #         if adjacent_cell_type in [BlockType.BEDROCK.value, BlockType.WOODER_DOOR.value, BlockType.SAFE.value]:
        #             continue
        #         adjacent_cell_name = PddlGameMapCellType.get_cell_object_name(adjacent_cell)
        #         fluent_to_value[(Predicate.adjacent_to_safe.name, adjacent_cell_name, cell_name)] = True

        return fluent_to_value


class Task:
    """ A task that the polycraft agent can aim to do """

    def __str__(self):
        return self.__class__.__name__

    def create_relevant_actions(self, world_state: PolycraftState, meta_model):
        """ Returns a list of actions for the agent to use when planning """
        raise NotImplementedError()

    def create_relevant_events(self, world_state: PolycraftState, meta_model):
        """ Returns a list of events for the agent to consider when planning"""
        raise NotImplementedError()

    def get_goals(self, world_state: PolycraftState, meta_model):
        """ Returns the goal to achieve """
        raise NotImplementedError()

    def get_metric(self, world_state: PolycraftState, meta_model):
        """ Returns the metric the planner seeks to optimize """
        raise NotImplementedError()

    def get_planner_heuristic(self, world_state: PolycraftState, metamodel):
        """ Returns the heuristic to be used by the planner"""
        raise NotImplementedError()

    def get_relevant_types(self, world_state: PolycraftState, meta_model):
        """ Returns a list of actions for the agent to use when planning """
        raise NotImplementedError()

    def get_relevant_predicates(self, world_state: PolycraftState, meta_model):
        """ Returns a list of actions for the agent to use when planning """
        raise NotImplementedError()

    def get_relevant_functions(self, world_state: PolycraftState, meta_model):
        """ Returns a list of actions for the agent to use when planning """
        raise NotImplementedError()

    def get_type_for_cell(self, cell_attr, meta_model) -> PddlGameMapCellType:
        """ Return a PddlGameMapCellType appropriate to generate objects representing this game map cell"""
        raise NotImplementedError()

    def is_done(self, state: PolycraftState) -> bool:
        """ Checks if the task has been succesfully completed """
        raise NotImplementedError()

    def is_feasible(self, state: PolycraftState) -> bool:
        """ Checks if the task can be achived in the current state """
        return True


class PddlPolycraftAction(PolycraftAction):
    """ Wrapper for Polycraft World Action that also stores the grounded pddl action that corresponds to this action """

    def __init__(self, poly_action, pddl_name, binding):
        super().__init__()
        self.poly_action = poly_action
        self.binding = binding
        self.pddl_name = pddl_name

    def is_success(self, result: dict):
        return self.poly_action.is_success(result)

    def __str__(self):
        if len(self.binding) > 0:
            params_str = " ".join([f"{k}={v}" for k, v in self.binding.items()])
            return f"<({self.pddl_name} {params_str}) success={self.success}>"
        else:
            return f"<({self.pddl_name}) success={self.success}>"

    def do(self, state: PolycraftState, env) -> dict:
        result = self.poly_action.do(state, env)
        self.success = self.poly_action.success
        return result

    def can_do(self, state: PolycraftState, env) -> bool:
        return self.poly_action.can_do(state, env)


class PddlPolycraftActionGenerator():
    """ An object that bridges between pddl actions and polycraft actions"""

    def __init__(self, pddl_name):
        self.pddl_name = pddl_name  # The name of this PDDL action

    """ A class representing a PDDL+ action in polycraft """

    def to_pddl(self, meta_model: MetaModel) -> PddlPlusWorldChange:
        """ This method should be implemented by sublcasses and output a string representation of the corresponding PDDL+ action """
        raise NotImplementedError()

    def to_polycraft(self, parameter_binding: dict) -> PolycraftAction:
        """ This method should be implemented by sublcasses and output a string representation of the corresponding PDDL+ action """
        raise NotImplementedError()

    def to_pddl_polycraft(self, parameter_binding: dict) -> PddlPolycraftAction:
        return PddlPolycraftAction(poly_action=self.to_polycraft(parameter_binding),
                                   pddl_name=self.pddl_name,
                                   binding=parameter_binding)


class PolycraftMetaModel(MetaModel):
    """ Sets the default meta-model"""

    def __init__(self, active_task: Task = None):
        super().__init__(docker_path=settings.POLYCRAFT_PLANNING_DOCKER_PATH,
                         domain_file_name="polycraft_domain.pddl",
                         delta_t=settings.POLYCRAFT_DELTA_T,
                         metric='minimize(total-time)',
                         repairable_constants=[
                             'break_log_outcome_num',
                             'break_platinum_outcome_num',
                             'break_diamond_outcome_num',
                             'collect_sap_outcome_num'],
                         repair_deltas=[
                             1, 1, 1, 1
                         ],
                         constant_numeric_fluents={
                             'break_log_outcome_num': 2,
                             'break_platinum_outcome_num': 1,
                             'break_diamond_outcome_num': 9,
                             'collect_sap_outcome_num': 1
                         },
                         constant_boolean_fluents={})

        self.domain_name = "polycraft"
        self.problem_name = "polycraft_prob"

        # Maps a cell type to what we get if we break it. The latter is in the form of a pair (item type, quantity).
        self.break_block_to_outcome = dict()
        self.break_block_to_outcome[BlockType.LOG.value] = (ItemType.LOG.value,
                                                            self.constant_numeric_fluents['break_log_outcome_num'])
        self.break_block_to_outcome[BlockType.BLOCK_OF_PLATINUM.value] = (ItemType.BLOCK_OF_PLATINUM.value,
                                                                          self.constant_numeric_fluents['break_platinum_outcome_num'])
        self.break_block_to_outcome[BlockType.DIAMOND_ORE.value] = (ItemType.DIAMOND.value,
                                                                    self.constant_numeric_fluents['break_diamond_outcome_num'])

        # Maps a cell type to what we get if we collect from it. The latter is in the form of a pair (item type, quantity).
        self.collect_block_to_outcome = dict()
        self.collect_block_to_outcome[BlockType.TREE_TAP.value] = (ItemType.SACK_POLYISOPRENE_PELLETS.value,
                                                                   self.constant_numeric_fluents['collect_sap_outcome_num'])
        self.collect_block_to_outcome[BlockType.PLASTIC_CHEST.value] = (ItemType.KEY.value, 1)

        # List of cell types that require an iron pickaxe to break
        self.needs_iron_pickaxe = list()
        self.needs_iron_pickaxe.append(BlockType.BLOCK_OF_PLATINUM.value)
        self.needs_iron_pickaxe.append(BlockType.DIAMOND_ORE.value)

        # List of items that one may select
        self.selectable_items = list()
        self.selectable_items.append(ItemType.IRON_PICKAXE.value)
        self.selectable_items.append(ItemType.KEY.value)

        # Assign indices to block types and item types
        self.block_type_to_idx = dict()
        type_idx = 0
        for block_type in BlockType:
            name = block_type.value
            self.block_type_to_idx[name] = type_idx
            type_idx = type_idx + 1

        self.item_type_to_idx = dict()
        type_idx = 0
        for item_type in ItemType:
            name = item_type.value
            self.item_type_to_idx[name] = type_idx
            type_idx = type_idx + 1

        self.active_task = active_task

    def introduce_novel_block_type(self, block_type):
        if block_type in self.block_type_to_idx:
            logger.info(f"Block type {block_type} already known")
            return
        type_idx = max(self.block_type_to_idx.values()) + 1
        self.block_type_to_idx[block_type] = type_idx
        self.break_block_to_outcome[block_type] = (block_type, 0)
        # Assume unknown object creates items, but set initial number to 0
        fluent_name = 'break_' + self._convert_element_naming(block_type) + '_outcome_num'
        self.introduce_novel_inventory_item_type(block_type, False)
        self.constant_numeric_fluents[fluent_name] = 0
        self.repairable_constants.append(fluent_name)
        self.repair_deltas.append(1)
        # Assume new item is not collectable

    def introduce_novel_inventory_item_type(self, item_type, selectable=True):
        """ Introduce new item type."""
        if item_type in self.item_type_to_idx:
            logger.info(f"Item type {item_type} already known")
            return
        type_idx = max(self.item_type_to_idx.values()) + 1
        self.item_type_to_idx[item_type] = type_idx

        if item_type not in self.selectable_items and selectable:
            self.selectable_items.append(item_type)  # Assume unknown item is selectable unless told otherwise

    def introduce_novel_entity_type(self, entity_type):
        """ Introduce novel entity type"""
        logger.info(f"Currently ignoring novel entity type {entity_type}")

    def set_active_task(self, task: Task):
        """ Sets the active task for which to create PDDL domains and problems """
        self.active_task = task

    def create_pddl_domain(self, world_state: PolycraftState) -> PddlPlusDomain:
        """ Create a PDDL+ domain for the given observed state """
        # domain_file = "{}/{}".format(str(self.docker_path), "polycraft_domain_template.pddl")
        # domain_parser = PddlDomainParser()
        # pddl_domain = PddlPlusDomain()

        pddl_domain = PddlPlusDomain()
        pddl_domain.name = "polycraft"
        pddl_domain.requirements = [":typing", ":disjunctive-preconditions", ":fluents", ":negative-preconditions"]

        # Add object types
        for object_type in self.active_task.get_relevant_types(world_state, self):
            pddl_domain.types.append(object_type.name)

        # Add predicates
        for predicate_as_list in self.active_task.get_relevant_predicates(world_state, self):
            pddl_domain.predicates.append(predicate_as_list)

        # Add functions
        for function_as_list in self.active_task.get_relevant_functions(world_state, self):
            pddl_domain.functions.append(function_as_list)

        for item in self.item_type_to_idx:
            pddl_domain.functions.append([f"count_{item}", ])

        # Add actions
        for action_generator in self.get_action_generators(world_state):
            pddl_domain.actions.append(action_generator.to_pddl(self))

        # Add events
        for event_generator in self.active_task.create_relevant_events(world_state, self):
            pddl_domain.events.append(event_generator.to_pddl(self))

        self._convert_polycraft_naming_in_domain(pddl_domain)
        return pddl_domain

    def get_action_generators(self, state):
        return self.active_task.create_relevant_actions(state, self)

    def _convert_polycraft_naming_in_domain(self, pddl_domain: PddlPlusDomain):
        """ Change the elements in the domain so that they fit the pddl convention of not using ":" """
        for i, pddl_element in enumerate(pddl_domain.functions):
            pddl_domain.functions[i] = self._convert_element_naming(pddl_element)
        for i, pddl_element in enumerate(pddl_domain.predicates):
            pddl_domain.predicates[i] = self._convert_element_naming(pddl_element)
        for i, pddl_element in enumerate(pddl_domain.constants):
            pddl_domain.constants[i] = self._convert_element_naming(pddl_element)
        for pddl_element in pddl_domain.actions:
            self._convert_world_change_naming(pddl_element)
        for pddl_element in pddl_domain.processes:
            self._convert_world_change_naming(pddl_element)
        for pddl_element in pddl_domain.events:
            self._convert_world_change_naming(pddl_element)

    def _convert_polycraft_naming_in_problem(self, pddl_problem: PddlPlusProblem):
        """ Change the elements in the problem so that they fit the pddl convention of not using ":" """
        for i, init_elment in enumerate(pddl_problem.init):
            pddl_problem.init[i] = self._convert_element_naming(init_elment)
        for i, goal_elment in enumerate(pddl_problem.goal):
            pddl_problem.goal[i] = self._convert_element_naming(goal_elment)

    def _convert_world_change_naming(self, world_change: PddlPlusWorldChange):
        """ Change the elements in this world change so that they fit the pddl convention of not using ":" """
        world_change.name = self._convert_element_naming(world_change.name)
        self._convert_element_naming(world_change.effects)
        self._convert_element_naming(world_change.preconditions)
        self._convert_element_naming(world_change.parameters)

    def _convert_element_naming(self, element):
        """ Recursive replace of : to _ to change polycraft naming to pddl """
        if type(element) == str:
            return element.replace(":", "_")
        if type(element) != list:
            return element
        for i, element_part in enumerate(element):
            element[i] = self._convert_element_naming(element_part)
        return element

    def _should_ignore_cell(self, cell: str, world_state: PolycraftState):
        """ An optimization step: identify cells that are not needed for the problem solving """
        cell_types_to_ignore = [BlockType.BEDROCK.value]
        known_cells = world_state.get_known_cells()
        type_str = known_cells[cell]['name']
        if type_str in cell_types_to_ignore:
            return True

        # Keep all non-air blocks
        if type_str != BlockType.AIR.value:
            return False

        # Keep all cells that occupy an entity (E.g., trader)
        for entity_id, entity_attr in world_state.entities.items():
            if cell == coordinates_to_cell(entity_attr["pos"]):
                return False

        # Ignore air cell that all its neighbors are also air cells
        all_neighbors_air = True
        for neighbor_cell in get_adjacent_cells(cell):
            if neighbor_cell in known_cells:
                neighbor_type = known_cells[neighbor_cell]['name']
                if neighbor_type != BlockType.AIR.value:
                    all_neighbors_air = False
                    break
        if all_neighbors_air:
            return True
        else:
            return False

    def create_pddl_problem(self, world_state: PolycraftState):
        """ Creates a PDDL problem file in which the given world state is the initial state """
        pddl_problem = PddlPlusProblem()
        pddl_problem.domain = self.domain_name
        pddl_problem.name = self.problem_name
        pddl_problem.metric = self.metric
        pddl_problem.objects = []
        pddl_problem.init = []
        pddl_problem.goal = []

        # A dictionary with global problem parameters
        problem_params = dict()
        problem_params["world_state"] = world_state

        # For efficiency reasons, we consider only a subset of the cells in the map
        known_cells = world_state.get_known_cells()
        active_cells = list()
        for cell, cell_attr in known_cells.items():
            # Pruning cells to gain efficiency
            if self._should_ignore_cell(cell, world_state) == False:
                active_cells.append(cell)
        problem_params["active_cells"] = active_cells

        # Add fluents for the active game map cells to the problem
        for cell in active_cells:
            cell_attr = known_cells[cell]
            type = self.active_task.get_type_for_cell(cell_attr, self)
            type.add_object_to_pddl((cell, cell_attr), pddl_problem, problem_params)

        # Add inventory items
        for item_type in self.item_type_to_idx.keys():
            count = world_state.count_items_of_type(item_type)
            pddl_problem.init.append(['=', [f"count_{item_type}", ], f"{count}"])

        # Add selected item
        select_item = world_state.get_selected_item()
        fluent_name = Function.selectedItem.name
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

        pddl_problem.init.append(["=", [Function.Steve_x.name, ], world_state.location["pos"][0]])
        pddl_problem.init.append(["=", [Function.Steve_z.name, ], world_state.location["pos"][2]])

        # Add goal and metric
        for goal in self.active_task.get_goals(world_state, self):
            pddl_problem.goal.append(goal)
        pddl_problem.metric = self.active_task.get_metric(world_state, self)
        self._convert_polycraft_naming_in_problem(pddl_problem)
        return pddl_problem

    def create_pddl_state(self, world_state: PolycraftState) -> PddlPlusState:
        """ Translate the given observed world state to a PddlPlusState object """

        pddl_state = PddlPlusState()
        # A dictionary with global problem parameters
        problem_params = dict()
        problem_params["world_state"] = world_state

        # For efficiency reasons, we consider only a subset of the cells in the map
        known_cells = world_state.get_known_cells()
        active_cells = list()
        for cell, cell_attr in known_cells.items():
            # Pruning cells to gain efficiency
            if not self._should_ignore_cell(cell, world_state):
                active_cells.append(cell)
        problem_params["active_cells"] = active_cells

        # Add fluents for the active game map cells to the problem
        for cell in active_cells:
            cell_attr = known_cells[cell]
            type = self.active_task.get_type_for_cell(cell_attr, self)
            type.add_object_to_state(pddl_state, (cell, cell_attr), problem_params)

        # Add inventory items
        for item_type in self.item_type_to_idx.keys():
            count = world_state.count_items_of_type(item_type)
            pddl_state.numeric_fluents[(f"count_{self._convert_element_naming(item_type)}",)] = count

        # Add selected item
        select_item = world_state.get_selected_item()
        fluent_name = (Function.selectedItem.name,)
        selected_item_idx = -1
        if select_item is not None and select_item in self.item_type_to_idx:
            selected_item_idx = self.item_type_to_idx[select_item]
            pddl_state.numeric_fluents[fluent_name] = selected_item_idx
        else:
            pddl_state.numeric_fluents[fluent_name] = -1

        # Add other entities
        for entity, entity_attr in world_state.entities.items():
            type_str = entity_attr["type"]
            if EntityType.TRADER.value == type_str:
                entity_cell = coordinates_to_cell(entity_attr["pos"])
                cell_name = PddlGameMapCellType.get_cell_object_name(entity_cell)
                pddl_state.boolean_fluents.add(f"trader_{entity}_at" + cell_name)

        pddl_state.numeric_fluents[(Function.Steve_x.name,)] = world_state.location["pos"][0]
        pddl_state.numeric_fluents[(Function.Steve_z.name,)] = world_state.location["pos"][2]

        return pddl_state

    def get_nyx_heuristic(self, world_state, meta_model):
        return self.active_task.get_planner_heuristic(world_state, meta_model)

    #
    # def _extract_landmarks(self,world_state: PolycraftState, pddl_problem:PddlPlusProblem, pddl_domain: PddlPlusDomain):
    #     goal = (ItemType.WOODEN_POGO_STICK.value,1)
    #     landmarks = set()
    #     active_landmarks = [goal]
    #     while len(landmarks)>0:
    #         landmark = landmarks.pop()
    #         landmarks.add(landmark)
    #
    #
    #
