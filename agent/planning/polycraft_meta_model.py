from agent.planning.meta_model import MetaModel
from agent.planning.polycraft_planning.actions import *
from agent.planning.polycraft_planning.polycaft_task_class import Task
from agent.planning.polycraft_planning.polycraft_pddl_objects_and_constants import PddlGameMapCellType, Function
from worlds.polycraft_world import *
from agent.planning.pddl_plus import *

logging.basicConfig(format='%(name)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PolycraftPDDL")
logger.setLevel(logging.DEBUG)


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
                             'break_diamond_outcome_num': 7,
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
                                                                          self.constant_numeric_fluents[
                                                                              'break_platinum_outcome_num'])
        self.break_block_to_outcome[BlockType.DIAMOND_ORE.value] = (ItemType.DIAMOND.value,
                                                                    self.constant_numeric_fluents[
                                                                        'break_diamond_outcome_num'])

        # Maps a cell type to what we get if we collect from it. The latter is in the form of a pair (item type, quantity).
        self.collect_block_to_outcome = dict()
        self.collect_block_to_outcome[BlockType.TREE_TAP.value] = (ItemType.SACK_POLYISOPRENE_PELLETS.value,
                                                                   self.constant_numeric_fluents[
                                                                       'collect_sap_outcome_num'])
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
        self.current_domain = {self.active_task: None}

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
        # if self.current_domain.get(self.active_task) is None:
            # First time creating a domain for this task - create from scratch
            # TODO: apply updates\MMOs already applied to other domains, then re-instate 'if domain for task exists'

        # TODO this code only for AAAI data; need to do this properly later
        self.break_block_to_outcome = dict()
        self.break_block_to_outcome[BlockType.LOG.value] = (ItemType.LOG.value,
                                                            self.constant_numeric_fluents['break_log_outcome_num'])
        self.break_block_to_outcome[BlockType.BLOCK_OF_PLATINUM.value] = (ItemType.BLOCK_OF_PLATINUM.value,
                                                                          self.constant_numeric_fluents[
                                                                              'break_platinum_outcome_num'])
        self.break_block_to_outcome[BlockType.DIAMOND_ORE.value] = (ItemType.DIAMOND.value,
                                                                    self.constant_numeric_fluents[
                                                                        'break_diamond_outcome_num'])

        # Maps a cell type to what we get if we collect from it. The latter is in the form of a pair (item type, quantity).
        self.collect_block_to_outcome = dict()
        self.collect_block_to_outcome[BlockType.TREE_TAP.value] = (ItemType.SACK_POLYISOPRENE_PELLETS.value,
                                                                   self.constant_numeric_fluents[
                                                                       'collect_sap_outcome_num'])


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
        # else:
        #     pddl_domain = self.current_domain[self.active_task]
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
            if not self._should_ignore_cell(cell, world_state):
                active_cells.append(cell)
        problem_params["active_cells"] = active_cells

        # Add fluents for the active game map cells to the problem
        for cell in active_cells:
            cell_attr = known_cells[cell]
            c_type = self.active_task.get_type_for_cell(cell_attr, self)
            c_type.add_object_to_problem(pddl_problem, (cell, cell_attr), problem_params)

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
            c_type = self.active_task.get_type_for_cell(cell_attr, self)
            c_type.add_object_to_state(pddl_state, (cell, cell_attr), problem_params)

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
