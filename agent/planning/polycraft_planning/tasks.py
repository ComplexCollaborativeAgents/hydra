from agent.planning.polycraft_meta_model import *
from agent.planning.nyx.abstract_heuristic import AbstractHeuristic
from agent.planning.polycraft_meta_model import PddlPolycraftActionGenerator
from agent.planning.polycraft_planning.actions import *
from worlds.polycraft_actions import PolySelectItem, PolyPlaceTreeTap, PolyCraftItem


class CreatePogoTask(Task):
    ''' A task that the polycraft agent can aim to do '''

    def get_relevant_types(self, world_state: PolycraftState, meta_model):
        ''' Returns a list of actions for the agent to use when planning '''
        return [PddlType.cell]

    def get_relevant_predicates(self, world_state: PolycraftState, meta_model):
        ''' Returns a list of actions for the agent to use when planning '''
        predicates = [Predicate.isAccessible.to_pddl(),
                      Predicate.adjacent.to_pddl()]

        for trader_id in set(world_state.trades.keys()):
            predicates.append([f"trader_{trader_id}_at", "?c", "-", "cell"])
        return predicates

    def get_relevant_functions(self, world_state: PolycraftState, meta_model):
        ''' Returns a list of actions for the agent to use when planning '''
        return [Function.cell_type.to_pddl(), Function.selectedItem.to_pddl()]

    def create_relevant_actions(self, world_state: PolycraftState, meta_model: PolycraftMetaModel) -> list:
        action_generators = []
        action_generators.extend(self._create_select_item_actions(meta_model))
        action_generators.extend(self._create_break_block_actions(meta_model))
        action_generators.extend(self._create_collect_actions(meta_model))
        action_generators.extend(self._create_craft_item_actions(world_state))
        action_generators.extend(self._create_trade_actions(world_state))

        action_generators.append(PddlPlaceTreeTapActionGenerator())
        action_generators.append(PddlCollectFromTreeTapActionGenerator())
        return action_generators

    def _create_select_item_actions(self, meta_model):
        # Create select item action generators
        pddl_actions = []
        for item_type in meta_model.selectable_items:
            pddl_actions.append(PddlSelectActionGenerator(item_type=item_type))
        return pddl_actions

    def _create_trade_actions(self, world_state):
        # Create trade actions generators
        pddl_actions = []
        for trader_id, trades in world_state.trades.items():
            for trade_idx, trade in enumerate(trades):
                pddl_actions.append(PddlTradeActionGenerator(trader_id, trade_idx, trade))
        return pddl_actions

    def _create_craft_item_actions(self, world_state):
        # Create crfat item (recipes) actions
        pddl_actions = []
        for recipe_idx, recipe in enumerate(world_state.recipes):
            craft_action = PolyCraftItem.create_action(recipe)
            if len(craft_action.recipe) == 9:  # Need a crafting table
                pddl_actions.append(PddlCraftActionGenerator(recipe_idx, recipe, needs_crafting_table=True))
            else:
                pddl_actions.append(PddlCraftActionGenerator(recipe_idx, recipe, needs_crafting_table=False))
        return pddl_actions

    def _create_collect_actions(self, meta_model):
        # Add collect item to outcome
        pddl_actions = []
        for block_type, outcome in meta_model.collect_block_to_outcome.items():
            if block_type == BlockType.TREE_TAP.value:
                continue  # Special treatment to collecting from tree tap
            item_type = outcome[0]
            quantity = outcome[1]
            pddl_actions.append(PddlCollectActionGenerator(block_type, item_type, quantity))
        return pddl_actions

    def _create_break_block_actions(self, meta_model):
        # Add blocks to break and get items
        pddl_actions = []
        for block_type, outcome in meta_model.break_block_to_outcome.items():
            item_type = outcome[0]
            quantity = outcome[1]
            if block_type in meta_model.needs_iron_pickaxe:
                needs_iron_pickaxe = True
            else:
                needs_iron_pickaxe = False
            pddl_actions.append(PddlBreakActionGenerator(block_type, item_type, quantity,
                                                         needs_iron_pickaxe=needs_iron_pickaxe))
        return pddl_actions

    def get_type_for_cell(self, cell_attr, meta_model) -> PddlGameMapCellType:
        ''' Return a PddlGameMapCellType appropriate to generate objects representing this game map cell'''
        type_str = cell_attr['name']
        if type_str not in meta_model.block_type_to_idx:
            logger.info("Unknown game map cell type: %s" % type_str)
            return PddlGameMapCellType(type_idx=-1)
        else:
            return PddlGameMapCellType(type_idx=meta_model.block_type_to_idx[type_str])

    def get_goals(self, world_state: PolycraftState, meta_model: PolycraftMetaModel):
        return [['>', [f"count_{ItemType.WOODEN_POGO_STICK.value}", ], "0"]]

    def create_relevant_events(self, world_state: PolycraftState, meta_model: PolycraftMetaModel):
        return [CellAccessibleEvent()]

    def get_metric(self, world_state: PolycraftState, meta_model: PolycraftMetaModel):
        return 'minimize(total-time)'

    def get_planner_heuristic(self, world_state: PolycraftState):
        ''' Returns the heuristic to be used by the planner'''
        return CraftPogoHeuristic(world_state)

    def is_done(self, state: PolycraftState) -> bool:
        ''' Checks if the task has been succesfully completed '''
        if state.count_items_of_type(ItemType.WOODEN_POGO_STICK.value) > 0:
            return True
        else:
            return False


class ExploreDoorTask(CreatePogoTask):
    def __init__(self, door_cell: str = None):
        self.door_cell = door_cell

    def is_done(self, state: PolycraftState) -> bool:
        ''' Checks if the task has been succesfully completed '''
        return is_steve_in_room(self.door_cell, state)

    def is_feasible(self, state: PolycraftState) -> bool:
        ''' Checks if the task can be achived in the current state '''
        return self.door_cell in state.get_type_to_cells()[BlockType.WOODER_DOOR.value]

    def get_relevant_types(self, world_state: PolycraftState, meta_model):
        ''' Returns a list of actions for the agent to use when planning '''
        return [PddlType.cell, PddlType.door_cell]

    def get_relevant_predicates(self, world_state: PolycraftState, meta_model):
        ''' Returns a list of actions for the agent to use when planning '''
        return [Predicate.isAccessible.to_pddl(),
                Predicate.adjacent.to_pddl(),
                Predicate.door_is_accessible.to_pddl(),
                Predicate.adjacent_to_door.to_pddl(),
                Predicate.open.to_pddl(),
                Predicate.passed_door.to_pddl()]

    def get_relevant_functions(self, world_state: PolycraftState, meta_model):
        ''' Returns a list of actions for the agent to use when planning '''
        return [Function.cell_type.to_pddl(),
                Function.selectedItem.to_pddl(),
                Function.door_cell_type.to_pddl()]

    def get_goals(self, world_state: PolycraftState, meta_model: PolycraftMetaModel):
        return [[Predicate.passed_door.name, PddlGameMapCellType.get_cell_object_name(self.door_cell)]]

    def get_planner_heuristic(self, world_state: PolycraftState):
        ''' Returns the heuristic to be used by the planner'''
        return OpenDoorHeuristic(self.door_cell)

    def create_relevant_actions(self, world_state: PolycraftState, meta_model: PolycraftMetaModel) -> list:
        action_generators = []
        action_generators.extend(self._create_select_item_actions(meta_model))
        action_generators.extend(self._create_break_block_actions(meta_model))
        action_generators.extend(self._create_collect_actions(meta_model))
        action_generators.extend([PddlMoveThroughDoorActionGenerator(),
                                  PddlUseDoorActionGenerator()])
        return action_generators

    def create_relevant_events(self, world_state: PolycraftState, meta_model: PolycraftMetaModel):
        return [CellAccessibleEvent(), DoorAccessibleEvent()]

    def get_type_for_cell(self, cell_attr, meta_model) -> PddlGameMapCellType:
        ''' Return a PddlGameMapCellType appropriate to generate objects representing this game map cell'''
        type_str = cell_attr['name']
        if type_str not in meta_model.block_type_to_idx:
            logger.info("Unknown game map cell type: %s" % type_str)
            return PddlGameMapCellType(type_idx=-1)
        else:
            type_idx = meta_model.block_type_to_idx[type_str]
            if type_str == BlockType.WOODER_DOOR.value:
                return PddlDoorCellType(type_idx=type_idx)
            else:
                return PddlGameMapCellType(type_idx=type_idx)


class CollectFromSafeTask(CreatePogoTask):
    ''' Task includes obtaining a key if none exists, going to the safe, opening it with the key and collecting what's in'''

    def __init__(self, safe_cell: str):
        self.safe_cell = safe_cell

    def is_feasible(self, state: PolycraftState) -> bool:
        ''' Checks if the task can be achived in the current state '''
        return self.safe_cell in state.get_type_to_cells()[BlockType.SAFE.value]

    def is_done(self, state: PolycraftState) -> bool:
        return False  # TODO: Mark in some way which safes have been collected

    def get_goals(self, world_state: PolycraftState, meta_model: PolycraftMetaModel):
        return [[Predicate.safe_collected.name, PddlGameMapCellType.get_cell_object_name(self.safe_cell)]]

    def get_relevant_types(self, world_state: PolycraftState, meta_model):
        ''' Returns a list of actions for the agent to use when planning '''
        return [PddlType.cell, PddlType.safe_cell]

    def get_relevant_predicates(self, world_state: PolycraftState, meta_model):
        ''' Returns a list of actions for the agent to use when planning '''
        return [Predicate.isAccessible.to_pddl(),
                Predicate.adjacent.to_pddl(),
                Predicate.safe_is_accessible.to_pddl(),
                Predicate.adjacent_to_safe.to_pddl(),
                Predicate.safe_collected.to_pddl(),
                Predicate.safe_open.to_pddl()]

    def get_relevant_functions(self, world_state: PolycraftState, meta_model):
        ''' Returns a list of actions for the agent to use when planning '''
        return [Function.cell_type.to_pddl(), Function.selectedItem.to_pddl()]

    def get_planner_heuristic(self, world_state: PolycraftState):
        ''' Returns the heuristic to be used by the planner'''
        return CollectFromSafeHeuristic(self.safe_cell)

    def create_relevant_actions(self, world_state: PolycraftState, meta_model: PolycraftMetaModel) -> list:
        action_generators = []
        action_generators.extend(self._create_select_item_actions(meta_model))
        action_generators.extend(self._create_break_block_actions(meta_model))
        action_generators.extend(self._create_collect_actions(meta_model))

        action_generators.append(PddlOpenSafeAndCollectGenerator())
        return action_generators

    def get_type_for_cell(self, cell_attr, meta_model) -> PddlGameMapCellType:
        ''' Return a PddlGameMapCellType appropriate to generate objects representing this game map cell'''
        type_str = cell_attr['name']
        if type_str not in meta_model.block_type_to_idx:
            logger.info("Unknown game map cell type: %s" % type_str)
            return PddlGameMapCellType(type_idx=-1)
        else:
            type_idx = meta_model.block_type_to_idx[type_str]
            if type_str == BlockType.SAFE.value:
                return PddlSafeCellType(type_idx=type_idx)
            else:
                return PddlGameMapCellType(type_idx=type_idx)

    def get_goals(self, world_state: PolycraftState, meta_model: PolycraftMetaModel):
        return [[Predicate.safe_collected.name, PddlGameMapCellType.get_cell_object_name(self.safe_cell)]]

    def create_relevant_events(self, world_state: PolycraftState, meta_model: PolycraftMetaModel):
        return [CellAccessibleEvent(), SafeAccessibleEvent()]

    def get_metric(self, world_state: PolycraftState, meta_model: PolycraftMetaModel):
        return 'minimize(total-time)'

    def add_cell_objects_to_problem(self, cell_obj: tuple, pddl_problem: PddlPlusProblem, problem_params: dict, ):
        ''' Adds the objects relevant for this cell to the given pddl_problem  '''
        cell, cell_attr = cell_obj
        type_str = cell_attr['name']
        if type_str not in self.block_type_to_idx:
            logger.info("Unknown game map cell type: %s" % type_str)
            type = PddlGameMapCellType(type_idx=-1)
        elif type_str == BlockType.SAFE.value:
            type = PddlSafeCellType(type_idx=self.block_type_to_idx[type_str])
        else:
            type = PddlGameMapCellType(type_idx=self.block_type_to_idx[type_str])
        type.add_object_to_pddl((cell, cell_attr), pddl_problem, problem_params)


class MakeCellAccessibleTask(CreatePogoTask):
    ''' TODO: Complete this '''

    def __init__(self, cell: str = None):
        self.cell = cell

    def get_goals(self, world_state: PolycraftState, meta_model: PolycraftMetaModel):
        return [[Predicate.isAccessible.name, PddlGameMapCellType.get_cell_object_name(self.cell)]]

    def get_planner_heuristic(self, world_state: PolycraftState):
        ''' Returns the heuristic to be used by the planner'''
        return MakeCellAccessibleHeuristic(self.cell)

    def is_done(self, state: PolycraftState) -> bool:
        ''' Checks if the task has been succesfully completed '''
        if state.get_known_cells()[self.cell][Predicate.isAccessible.name]:
            return True
        else:
            return False


### Heuristic functions for tasks

class CraftPogoHeuristic(AbstractHeuristic):
    ''' Heuristic for polycraft to be used by the Nyx planner '''

    def __init__(self, world_state: PolycraftState):
        # Get pogo recipe
        pogo_recipe = world_state.get_recipe_for(ItemType.WOODEN_POGO_STICK.value)
        pogo_ingredients = get_ingredients_for_recipe(pogo_recipe)
        self.ingredients = list()
        for item_type, quantity in pogo_ingredients.items():
            pddl_item_type = f"count_{item_type.replace(':', '_')}"
            self.ingredients.append((pddl_item_type, quantity))

    def evaluate(self, node):
        # Check if have ingredients of pogo stick
        pogo_count = node.state_vars["['count_polycraft_wooden_pogo_stick']"]
        if pogo_count > 0:
            return 0

        h_value = 1
        for fluent, quantity in self.ingredients:
            delta = quantity - node.state_vars[f"['{fluent}']"]
            if delta > 0:
                h_value = h_value + delta
        node.h = h_value
        return h_value


class OpenDoorHeuristic(AbstractHeuristic):
    ''' Heuristic for polycraft to be used by the Nyx planner when solving an open door task'''

    def __init__(self, door_cell: str):
        super().__init__()
        self.door_cell = door_cell
        pddl_door_cell_name = PddlGameMapCellType.get_cell_object_name(door_cell)
        self._is_door_accessible_var = f"['{Predicate.door_is_accessible.name}', '{pddl_door_cell_name}']"
        self._is_open_var = f"['{Predicate.open.name}', '{pddl_door_cell_name}']"
        self._passed_door_var = f"['{Predicate.passed_door.name}', '{pddl_door_cell_name}']"

    def evaluate(self, node):
        # Check if have ingredients of pogo stick
        if node.state_vars[self._passed_door_var]:
            return 0
        if node.state_vars[self._is_open_var]:
            return 1
        if node.state_vars[self._is_door_accessible_var]:
            return 2
        else:
            return 3  # Can improve this


class CollectFromSafeHeuristic(AbstractHeuristic):
    ''' Heuristic for polycraft to be used by the Nyx planner when solving an collect from safe task'''

    def __init__(self, safe_cell: str):
        super().__init__()
        self.safe_cell = safe_cell

    def evaluate(self, node):
        return 0  # TODO: IMPROVE THIS


class MakeCellAccessibleHeuristic(AbstractHeuristic):
    ''' Heuristic for polycraft to be used by the Nyx planner when solving an open door task'''

    def __init__(self, cell: str):
        super().__init__()
        self.cell = PddlGameMapCellType.get_cell_object_name(cell)
        self.pddl_cell_name = {PddlGameMapCellType.get_cell_object_name(self.cell)}
        self._is_accessible_var = f"['{Predicate.isAccessible.name}', '{pddl_cell_name}']"

    def evaluate(self, node):
        if node.state_vars[self._is_accessible_var]:
            return 0
        if node.state_vars[self.pddl_cell_name] != BlockType.AIR:
            return 1
        else:
            return 3  # Can improve this


########### Actions

###### PDDL ACTIONS, PROCESSES, AND EVENTS

class PddlPlaceTreeTapActionGenerator(PddlPolycraftActionGenerator):
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

    def __init__(self):
        super().__init__("place_tree_tap")

    def to_pddl(self, meta_model: MetaModel) -> PddlPlusWorldChange:
        pddl_action = PddlPlusWorldChange(WorldChangeTypes.action)
        pddl_action.name = self.pddl_name
        pddl_action.parameters.append(["?at", "-", "cell", "?near_to", "-", "cell"])

        pddl_action.preconditions.append([">=", [f"count_{ItemType.TREE_TAP.value}", ], "1"])
        log_idx = meta_model.block_type_to_idx[BlockType.LOG.value]
        pddl_action.preconditions.append(["=", ["cell_type", "?near_to"], f"{log_idx}"])
        air_idx = meta_model.block_type_to_idx[BlockType.AIR.value]
        pddl_action.preconditions.append(["=", ["cell_type", "?at"], f"{air_idx}"])
        pddl_action.preconditions.append(["adjacent", "?at", "?near_to"])
        pddl_action.preconditions.append(["isAccessible", "?at"])
        pddl_action.preconditions.append(["isAccessible", "?near_to"])

        pddl_action.effects.append(["decrease", [f"count_{ItemType.TREE_TAP.value}", ], "1"])
        tree_tap_idx = meta_model.block_type_to_idx[BlockType.TREE_TAP.value]
        pddl_action.effects.append(["assign", ["cell_type", "?at"], f"{tree_tap_idx}"])
        return pddl_action

    def to_polycraft(self, parameter_binding: dict) -> PolycraftAction:
        return TeleportToAndPlaceTreeTap(parameter_binding["?at"])


class PddlCollectActionGenerator(PddlPolycraftActionGenerator):
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

    def __init__(self, collect_from_block_type: str, item_type_to_collect: str, quantity):
        super().__init__(f"collect_{item_type_to_collect}_from_{collect_from_block_type}")
        self.item_type_to_collect = item_type_to_collect
        self.collect_from_block_type = collect_from_block_type
        self.quantity = quantity

    def to_pddl(self, meta_model: MetaModel) -> PddlPlusWorldChange:
        pddl_action = PddlPlusWorldChange(WorldChangeTypes.action)
        pddl_action.name = self.pddl_name
        pddl_action.parameters.append(["?c", "-", "cell"])

        cell_type_idx = meta_model.block_type_to_idx[self.collect_from_block_type]
        pddl_action.preconditions.append(["=", ["cell_type", "?c"], f"{cell_type_idx}"])
        pddl_action.preconditions.append(["isAccessible", "?c"])

        pddl_action.effects.append(["increase", [f"count_{self.item_type_to_collect}", ], str(self.quantity)])
        return pddl_action

    def to_polycraft(self, parameter_binding: dict) -> PolycraftAction:
        return TeleportToAndCollect(parameter_binding['?c'])


class PddlCollectFromTreeTapActionGenerator(PddlPolycraftActionGenerator):
    ''' An action corresponding to collecting an item from a cell (COLLECT in the Polycraft API)
    (:action collect_from_tree_tap
        :parameters (?c - cell near_to? - cell)
        :precondition( and
            (cell_type c? {tree_tap})
            (cell_type near_to? {log})
            (adjacent c? near_to?)
            (isAccessible ?c)
        )
        :effect( and
            (increase(count_{item_type}) 1)
        )
    )
    '''

    def __init__(self):
        super().__init__(f"collect_from_tree_tap")

    def to_pddl(self, meta_model: MetaModel) -> PddlPlusWorldChange:
        pddl_action = PddlPlusWorldChange(WorldChangeTypes.action)
        pddl_action.name = self.pddl_name
        pddl_action.parameters.append(["?c", "-", "cell", "?near_to", "-", "cell"])

        tree_tap_idx = meta_model.block_type_to_idx[BlockType.TREE_TAP.value]
        pddl_action.preconditions.append(["=", ["cell_type", "?c"], f"{tree_tap_idx}"])
        pddl_action.preconditions.append([Predicate.adjacent.name, "?c", "?near_to"])
        log_idx = meta_model.block_type_to_idx[BlockType.LOG.value]
        pddl_action.preconditions.append(["=", ["cell_type", "?near_to"], f"{log_idx}"])
        pddl_action.preconditions.append([Predicate.isAccessible.name, "?c"])

        sack_of_pellets_per_collect = meta_model.collect_block_to_outcome[BlockType.TREE_TAP.value][1]
        pddl_action.effects.append(["increase", [f"count_{ItemType.SACK_POLYISOPRENE_PELLETS.value}", ],
                                    str(sack_of_pellets_per_collect)])
        return pddl_action

    def to_polycraft(self, parameter_binding: dict) -> PolycraftAction:
        return TeleportToAndCollect(parameter_binding['?c'])


class PddlBreakActionGenerator(PddlPolycraftActionGenerator):
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

    def __init__(self, block_type_to_break: str, item_type_to_collect: str, items_per_block: int,
                 needs_iron_pickaxe=False):
        super().__init__(f"break_{block_type_to_break}")
        self.block_type_to_break = block_type_to_break
        self.item_type_to_collect = item_type_to_collect
        self.items_per_block = items_per_block
        self.needs_iron_pickaxe = needs_iron_pickaxe

    def to_pddl(self, meta_model: MetaModel) -> PddlPlusWorldChange:
        pddl_action = PddlPlusWorldChange(WorldChangeTypes.action)
        pddl_action.name = self.pddl_name
        pddl_action.parameters.append(["?c", "-", "cell"])

        cell_type_idx = meta_model.block_type_to_idx[self.block_type_to_break]
        pddl_action.preconditions.append(["=", ["cell_type", "?c"], f"{cell_type_idx}"])
        if self.needs_iron_pickaxe:
            iron_pickaxe_idx = meta_model.item_type_to_idx[ItemType.IRON_PICKAXE.value]
            pddl_action.preconditions.append(["=", ["selectedItem", ], f"{iron_pickaxe_idx}"])
        pddl_action.preconditions.append(["isAccessible", "?c"])

        pddl_action.effects.append(["increase", [f"count_{self.item_type_to_collect}", ], str(self.items_per_block)])
        air_cell_idx = meta_model.block_type_to_idx[BlockType.AIR.value]
        pddl_action.effects.append(["assign", ["cell_type", "?c"], f"{air_cell_idx}"])
        return pddl_action

    def to_polycraft(self, parameter_binding: dict) -> PolycraftAction:
        return TeleportToBreakAndCollect(parameter_binding['?c'])


class PddlSelectActionGenerator(PddlPolycraftActionGenerator):
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
        super().__init__(f"select_{item_type}")
        self.item_type_name = item_type

    def to_pddl(self, meta_model: MetaModel) -> PddlPlusWorldChange:
        pddl_action = PddlPlusWorldChange(WorldChangeTypes.action)
        pddl_action.name = self.pddl_name

        item_type_idx = meta_model.item_type_to_idx[self.item_type_name]
        pddl_action.preconditions.append(["not", ["=", ["selectedItem", ], f"{item_type_idx}"]])
        pddl_action.preconditions.append([">=", [f"count_{self.item_type_name}", ], "1"])

        pddl_action.effects.append(
            ["assign", ["selectedItem", ], f"{item_type_idx}"])
        return pddl_action

    def to_polycraft(self, parameter_binding: dict) -> PolycraftAction:
        return PolySelectItem(self.item_type_name)


class PddlUseDoorActionGenerator(PddlPolycraftActionGenerator):
    ''' An action for using a dor
        ; Use door (USE)
        (:action select_{item_type.name}
            :precondition (and
                (>= (count_{item_type.name}) 1)
            )
            :effect (and
                (assign (selectedItem) {item_type.value})
            )
        )
    '''

    def __init__(self):
        super().__init__(f"use_door")

    def to_pddl(self, meta_model: MetaModel) -> PddlPlusWorldChange:
        pddl_action = PddlPlusWorldChange(WorldChangeTypes.action)
        pddl_action.name = self.pddl_name
        pddl_action.parameters.append(["?c", "-", PddlType.door_cell.name])

        pddl_action.preconditions.append([Predicate.door_is_accessible.name, "?c"])
        pddl_action.preconditions.append(["not", [Predicate.open.name, "?c"]])
        pddl_action.effects.append([Predicate.open.name, "?c"])
        return pddl_action

    def to_polycraft(self, parameter_binding: dict) -> PolycraftAction:
        return OpenDoor(parameter_binding["?c"])


class PddlOpenSafeAndCollectGenerator(PddlPolycraftActionGenerator):
    ''' An action generator for opening a safe and collecting it '''

    def __init__(self):
        super().__init__(f"open_safe_and_collect")

    def to_pddl(self, meta_model: MetaModel) -> PddlPlusWorldChange:
        pddl_action = PddlPlusWorldChange(WorldChangeTypes.action)
        pddl_action.name = self.pddl_name
        pddl_action.parameters.append(["?c", "-", PddlType.safe_cell.name])

        pddl_action.preconditions.append([">=", [f"count_{ItemType.KEY.value}"], "1"])
        pddl_action.preconditions.append([Predicate.safe_is_accessible.name, "?c"])

        pddl_action.effects.append([Predicate.safe_collected.name, "?c"])

        return pddl_action

    def to_polycraft(self, parameter_binding: dict) -> PolycraftAction:
        return OpenSafeAndCollect(parameter_binding["?c"])


class PddlTradeActionGenerator(PddlPolycraftActionGenerator):
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

    def __init__(self, trader_id: str, trade_idx: int, trade):
        super().__init__(f"trade_recipe_{trader_id}_{trade_idx}")
        self.trader_id = trader_id
        self.trade_idx = trade_idx
        self.trade = trade

    def to_pddl(self, meta_model: MetaModel) -> PddlPlusWorldChange:
        pddl_action = PddlPlusWorldChange(WorldChangeTypes.action)
        pddl_action.name = self.pddl_name
        pddl_action.parameters.append(["?trader_loc", "-", "cell"])

        for input in self.trade["inputs"]:
            item_type = input['Item']
            quantity = input['stackSize']
            pddl_action.preconditions.append([">=", [f"count_{item_type}"], str(quantity)])
            pddl_action.effects.append(["decrease", [f"count_{item_type}"], str(quantity)])

        pddl_action.preconditions.append([f"trader_{self.trader_id}_at", "?trader_loc"])

        pddl_action.preconditions.append(["isAccessible", "?trader_loc"])

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
        pddl_event.parameters.append(["?c1", "-", PddlType.cell.name, "?c2", "-", PddlType.cell.name])
        pddl_event.preconditions.append(["not", [Predicate.isAccessible.name, "?c2"]])
        pddl_event.preconditions.append([Predicate.adjacent.name, "?c1", "?c2"])
        air_idx = meta_model.block_type_to_idx[BlockType.AIR.value]
        pddl_event.preconditions.append(["=", [Function.cell_type.name, "?c1"], f"{air_idx}"])
        pddl_event.preconditions.append([Predicate.isAccessible.name, "?c1"])

        pddl_event.effects.append([Predicate.isAccessible.name, "?c2"])
        return pddl_event


class DoorAccessibleEvent(PddlPolycraftEvent):
    def to_pddl(self, meta_model: MetaModel):
        pddl_event = PddlPlusWorldChange(WorldChangeTypes.event)
        pddl_event.name = "door_accessible"
        pddl_event.parameters.append(["?c1", "-", PddlType.cell.name,
                                      "?c2", "-", PddlType.door_cell.name])
        pddl_event.preconditions.append([Predicate.isAccessible.name, "?c1"])
        pddl_event.preconditions.append(["not", [Predicate.door_is_accessible.name, "?c2"]])
        air_idx = meta_model.block_type_to_idx[BlockType.AIR.value]
        pddl_event.preconditions.append(["=", [Function.cell_type.name, "?c1"], f"{air_idx}"])
        pddl_event.preconditions.append([Predicate.adjacent_to_door.name, "?c1", "?c2"])
        pddl_event.effects.append([Predicate.door_is_accessible.name, "?c2"])
        return pddl_event


class SafeAccessibleEvent(PddlPolycraftEvent):
    def to_pddl(self, meta_model: MetaModel):
        pddl_event = PddlPlusWorldChange(WorldChangeTypes.event)
        pddl_event.name = "safe_accessible"
        pddl_event.parameters.append(["?c1", "-", PddlType.cell.name,
                                      "?c2", "-", PddlType.safe_cell.name])
        pddl_event.preconditions.append([Predicate.isAccessible.name, "?c1"])
        pddl_event.preconditions.append(["not", [Predicate.safe_is_accessible.name, "?c2"]])
        air_idx = meta_model.block_type_to_idx[BlockType.AIR.value]
        pddl_event.preconditions.append(["=", [Function.cell_type.name, "?c1"], f"{air_idx}"])
        pddl_event.preconditions.append([Predicate.adjacent_to_safe.name, "?c1", "?c2"])
        pddl_event.effects.append([Predicate.safe_is_accessible.name, "?c2"])
        return pddl_event


class PddlCraftActionGenerator(PddlPolycraftActionGenerator):
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

    def __init__(self, recipe_idx: int, recipe, needs_crafting_table: bool = False):
        super().__init__(f"craft_recipe_{recipe_idx}")
        self.recipe_idx = recipe_idx
        self.recipe = recipe
        self.needs_crafting_table = needs_crafting_table

        # To make the domain file more human readable, compute a clearer pddl_action name
        action_name_suffix_elements = []
        for item_type, quantity in get_outputs_of_recipe(self.recipe).items():
            if quantity == 1:
                action_name_suffix_elements.append(item_type)
            else:
                action_name_suffix_elements.append(f"{quantity}_{item_type}")
        super().__init__(f"craft_recipe_{recipe_idx}_for_{'_and_'.join(action_name_suffix_elements)}")

    def to_pddl(self, meta_model: MetaModel) -> PddlPlusWorldChange:
        pddl_action = PddlPlusWorldChange(WorldChangeTypes.action)
        pddl_action.name = self.pddl_name

        for item_type, quantity in get_ingredients_for_recipe(self.recipe).items():
            pddl_action.preconditions.append([">=", [f"count_{item_type}"], str(quantity)])
            pddl_action.effects.append(["decrease", [f"count_{item_type}"], str(quantity)])

        if self.needs_crafting_table:  # TODO: Not robust to problems with multiple crafting tables
            pddl_action.parameters.append(["?from", "-", "cell"])
            crafting_table_idx = meta_model.block_type_to_idx[BlockType.CRAFTING_TABLE.value]
            pddl_action.preconditions.append(["=", ["cell_type", "?from", ], f"{crafting_table_idx}"])
            pddl_action.preconditions.append(["isAccessible", "?from"])

        for item_type, quantity in get_outputs_of_recipe(self.recipe).items():
            pddl_action.effects.append(["increase", [f"count_{item_type}"], str(quantity)])
        return pddl_action

    def to_polycraft(self, parameter_binding: dict) -> PolycraftAction:
        if self.needs_crafting_table:
            return TeleportToTableAndCraft(parameter_binding["?from"], self.recipe)
        else:
            return PolyCraftItem.create_action(self.recipe)


class PddlMoveThroughDoorActionGenerator(PddlPolycraftActionGenerator):
    ''' An action for moving through a door
        ; MOVE w w
        (:action move_through_door_{door_cell}
            :parameters (?cell - door_cell)
            :precondition( and
                (isAccessible ?cell)
                (open ?cell)
            )
        )
    '''

    def __init__(self):
        super().__init__("move_through_door")

    def to_pddl(self, meta_model: MetaModel) -> PddlPlusWorldChange:
        pddl_action = PddlPlusWorldChange(WorldChangeTypes.action)
        pddl_action.name = self.pddl_name
        pddl_action.parameters.append(["?cell", "-", PddlType.door_cell.name])
        pddl_action.preconditions.append([Predicate.door_is_accessible.name, "?cell"])
        pddl_action.preconditions.append([Predicate.open.name, "?cell"])
        pddl_action.effects.append([Predicate.passed_door.name, "?cell"])
        return pddl_action

    def to_polycraft(self, parameter_binding: dict) -> PolycraftAction:
        return MoveThroughDoor(parameter_binding["?cell"])


class PolycraftTask(enum.Enum):
    ''' The types of tasks this meta model object supports '''
    CRAFT_POGO = CreatePogoTask
    EXPLORE_DOOR = ExploreDoorTask
    MAKE_CELL_ACCESSIBLE = MakeCellAccessibleTask
    COLLECT_FROM_SAFE = CollectFromSafeTask

    def create_instance(self):
        ''' Create an instance of this task'''
        return self.value.__new__(self.value)
