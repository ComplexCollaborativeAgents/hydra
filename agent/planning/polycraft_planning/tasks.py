from math import ceil

from agent.planning.polycraft_meta_model import *
from agent.planning.nyx.abstract_heuristic import AbstractHeuristic
from agent.planning.polycraft_meta_model import PddlPolycraftActionGenerator
from agent.planning.polycraft_planning.actions import *
from worlds.polycraft_actions import PolySelectItem, PolyPlaceTreeTap, PolyCraftItem
import agent.planning.nyx.syntax.state as SearchState


class CreatePogoTask(Task):
    """ A task that the polycraft agent can aim to do """

    def get_relevant_types(self, world_state: PolycraftState, meta_model):
        """ Returns a list of actions for the agent to use when planning """
        return [PddlType.cell]

    def get_relevant_predicates(self, world_state: PolycraftState, meta_model):
        """ Returns a list of actions for the agent to use when planning """
        predicates = [Predicate.isAccessible.to_pddl()]

        for trader_id in set(world_state.trades.keys()):
            predicates.append([f"trader_{trader_id}_at", "?c", "-", "cell"])
        return predicates

    def get_relevant_functions(self, world_state: PolycraftState, meta_model):
        """ Returns a list of actions for the agent to use when planning """
        return [Function.cell_type.to_pddl(), Function.selectedItem.to_pddl(),
                Function.Steve_x.to_pddl(), Function.Steve_z.to_pddl(),
                Function.cell_x.to_pddl(), Function.cell_z.to_pddl()]

    def create_relevant_actions(self, world_state: PolycraftState, meta_model: PolycraftMetaModel) -> list:
        action_generators = []
        action_generators.extend(self._create_select_item_actions(meta_model))
        action_generators.extend(self._create_break_block_actions(meta_model))
        action_generators.extend(self._create_collect_actions(meta_model))
        action_generators.extend(self._create_craft_item_actions(world_state))
        action_generators.extend(self._create_trade_actions(world_state))
        action_generators.extend(self._create_place_tree_tap_actions(world_state, meta_model))
        action_generators.append(PddlTeleportActionGenerator())
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

    def _create_place_tree_tap_actions(self, world_state: PolycraftState, meta_model: PolycraftMetaModel):
        # If we're already hard-coding to existing trees, why not hard-code to the surrounding cells too?
        pddl_actions = []
        directions = [[1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1]]
        for cell, cell_attr in world_state.get_known_cells().items():
            if cell_attr["name"] == BlockType.LOG.value:
                coords = cell_to_coordinates(cell)
                for delta in directions:
                    air_cell_coords = [coord + d for coord, d in zip(coords, delta)]
                    air_cell = coordinates_to_cell(air_cell_coords)
                    pddl_actions.append(
                        PddlPlaceTreeTapActionGenerator(cell, air_cell))

        return pddl_actions

    def get_type_for_cell(self, cell_attr, meta_model) -> PddlGameMapCellType:
        """ Return a PddlGameMapCellType appropriate to generate objects representing this game map cell"""
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

    def get_planner_heuristic(self, world_state: PolycraftState, metamodel:PolycraftMetaModel):
        """ Returns the heuristic to be used by the planner"""
        return CraftPogoHeuristic(world_state, metamodel)

    def is_done(self, state: PolycraftState) -> bool:
        """ Checks if the task has been successfully completed """
        if state.count_items_of_type(ItemType.WOODEN_POGO_STICK.value) > 0:
            return True
        else:
            return False


class ExploreDoorTask(CreatePogoTask):
    def __init__(self, door_cell: str = None):
        self.door_cell = door_cell

    def is_done(self, state: PolycraftState) -> bool:
        """ Checks if the task has been succesfully completed """
        return is_steve_in_room(self.door_cell, state)

    def is_feasible(self, state: PolycraftState) -> bool:
        """ Checks if the task can be achived in the current state """
        return self.door_cell in state.get_type_to_cells()[BlockType.WOODER_DOOR.value]

    def get_relevant_types(self, world_state: PolycraftState, meta_model):
        """ Returns a list of actions for the agent to use when planning """
        return [PddlType.cell, PddlType.door_cell]

    def get_relevant_predicates(self, world_state: PolycraftState, meta_model):
        """ Returns a list of actions for the agent to use when planning """
        return [Predicate.isAccessible.to_pddl(),
                Predicate.door_is_accessible.to_pddl(),
                # Predicate.adjacent_to_door.to_pddl(),
                Predicate.open.to_pddl(),
                Predicate.passed_door.to_pddl()]

    def get_relevant_functions(self, world_state: PolycraftState, meta_model):
        """ Returns a list of actions for the agent to use when planning """
        return [Function.cell_type.to_pddl(),
                Function.selectedItem.to_pddl(),
                Function.door_cell_type.to_pddl(),
                Function.Steve_x.to_pddl(), Function.Steve_z.to_pddl(),
                Function.cell_x.to_pddl(), Function.cell_z.to_pddl()]

    def get_goals(self, world_state: PolycraftState, meta_model: PolycraftMetaModel):
        return [[Predicate.passed_door.name, PddlGameMapCellType.get_cell_object_name(self.door_cell)]]

    def get_planner_heuristic(self, world_state: PolycraftState, metamodel):
        """ Returns the heuristic to be used by the planner"""
        return OpenDoorHeuristic(self.door_cell)

    def create_relevant_actions(self, world_state: PolycraftState, meta_model: PolycraftMetaModel) -> list:
        action_generators = []
        action_generators.extend(self._create_select_item_actions(meta_model))
        action_generators.extend(self._create_break_block_actions(meta_model))
        action_generators.extend(self._create_collect_actions(meta_model))
        action_generators.extend([PddlMoveThroughDoorActionGenerator(),
                                  PddlUseDoorActionGenerator()])
        action_generators.append(PddlTeleportActionGenerator())
        return action_generators

    def create_relevant_events(self, world_state: PolycraftState, meta_model: PolycraftMetaModel):
        return [CellAccessibleEvent(), DoorAccessibleEvent()]

    def get_type_for_cell(self, cell_attr, meta_model) -> PddlGameMapCellType:
        """ Return a PddlGameMapCellType appropriate to generate objects representing this game map cell"""
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
    """ Task includes obtaining a key if none exists, going to the safe, opening it with the key and collecting what's in"""

    def __init__(self, safe_cell: str):
        self.safe_cell = safe_cell

    def is_feasible(self, state: PolycraftState) -> bool:
        """ Checks if the task can be achived in the current state """
        return self.safe_cell in state.get_type_to_cells()[BlockType.SAFE.value]

    def is_done(self, state: PolycraftState) -> bool:
        return False  # TODO: Mark in some way which safes have been collected

    def get_goals(self, world_state: PolycraftState, meta_model: PolycraftMetaModel):
        return [[Predicate.safe_collected.name, PddlGameMapCellType.get_cell_object_name(self.safe_cell)]]

    def get_relevant_types(self, world_state: PolycraftState, meta_model):
        """ Returns a list of actions for the agent to use when planning """
        return [PddlType.cell, PddlType.safe_cell]

    def get_relevant_predicates(self, world_state: PolycraftState, meta_model):
        """ Returns a list of actions for the agent to use when planning """
        return [Predicate.isAccessible.to_pddl(),
                Predicate.safe_is_accessible.to_pddl(),
                # Predicate.adjacent_to_safe.to_pddl(),
                Predicate.safe_collected.to_pddl(),
                Predicate.safe_open.to_pddl()]

    def get_relevant_functions(self, world_state: PolycraftState, meta_model):
        """ Returns a list of actions for the agent to use when planning """
        return [Function.cell_type.to_pddl(), Function.selectedItem.to_pddl(),
                Function.Steve_x.to_pddl(), Function.Steve_z.to_pddl(),
                Function.cell_x.to_pddl(), Function.cell_z.to_pddl()]

    def get_planner_heuristic(self, world_state: PolycraftState, metamodel):
        """ Returns the heuristic to be used by the planner"""
        return CollectFromSafeHeuristic(self.safe_cell)

    def create_relevant_actions(self, world_state: PolycraftState, meta_model: PolycraftMetaModel) -> list:
        action_generators = []
        action_generators.extend(self._create_select_item_actions(meta_model))
        action_generators.extend(self._create_break_block_actions(meta_model))
        action_generators.extend(self._create_collect_actions(meta_model))

        action_generators.append(PddlTeleportActionGenerator())

        action_generators.append(PddlOpenSafeAndCollectGenerator())
        return action_generators

    def get_type_for_cell(self, cell_attr, meta_model) -> PddlGameMapCellType:
        """ Return a PddlGameMapCellType appropriate to generate objects representing this game map cell"""
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
        """ Adds the objects relevant for this cell to the given pddl_problem  """
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
    """ TODO: Complete this """

    def __init__(self, cell: str = None):
        self.cell = cell

    def get_goals(self, world_state: PolycraftState, meta_model: PolycraftMetaModel):
        return [[Predicate.isAccessible.name, PddlGameMapCellType.get_cell_object_name(self.cell)]]

    def get_planner_heuristic(self, world_state: PolycraftState, metamodel):
        """ Returns the heuristic to be used by the planner"""
        return MakeCellAccessibleHeuristic(self.cell)

    def is_done(self, state: PolycraftState) -> bool:
        """ Checks if the task has been succesfully completed """
        if state.get_known_cells()[self.cell][Predicate.isAccessible.name]:
            return True
        else:
            return False


### Heuristic functions for tasks

class CraftPogoHeuristic(AbstractHeuristic):
    """ Heuristic for polycraft to be used by the Nyx planner """

    def __init__(self, world_state: PolycraftState, metamodel: PolycraftMetaModel):
        self.initial_state = world_state
        self.metamodel = metamodel

    # # The next few functions make a heuristic that should dominate '_only_count_once_heuristic', but empirically
    # #  performs less well.
    def _count_cells_of_type(self, node, type_str):
        """
        Returns the number of accessible cells of a given type, and the total number
        """
        type_id = self.metamodel.block_type_to_idx[type_str]
        accessible, total = 0, 0
        for fluent, value in node.state_vars.items():
            if fluent.find(Function.cell_type.name) > -1 and value == type_id:
                total += 1
                match = re.search('cell_\d{2}_\d{2}_\d{2}', fluent)
                acc_fluent = f"['{Predicate.isAccessible.name.lower()}', '{fluent[match.start():match.end()]}']"
                if node.state_vars[acc_fluent]:
                    accessible += 1
        return accessible, total

    def _get_inventory(self, search_node: SearchState):
        """
        Retrieves the player's current inventory (as a dictionary)
        """
        inventory = dict()
        for var, val in search_node.state_vars.items():
            if var[0:8] == "['count_":
                item_name = var[8:-2].replace('_', ":")
                inventory[item_name] = val
        return inventory

    def _count_recipe_prime_ingredients(self, item_type: str, inventory: dict):
        """
         Counts the total number of prime ingredients (not produced by a recipe) needed to create an item, less the
        items currently in the inventory.
        This is calculated recursively to avoid double-counting, e.g. if logs are needed on several branches of the
        recipe tree, we only count for the ones we still need.
        """
        if item_type in inventory.keys() and inventory[item_type] > 0:
            # Already have item
            return dict()
        recipe = self.initial_state.get_recipe_for(item_type)
        if recipe is None:
            # Not produced by recipe - mined\collected
            return {item_type: 1}
        # else
        items = dict()
        step1_ingredients = get_ingredients_for_recipe(recipe)
        for ingredient, quantity in step1_ingredients.items():
            if ingredient not in inventory.keys():
                inventory[ingredient] = 0
            if inventory[ingredient] >= quantity:
                # Have enough, skip this branch
                inventory[ingredient] -= quantity
                continue
            # Don't have enough of ingredient, update how much we need and get it
            quantity -= inventory[ingredient]
            inventory[ingredient] = 0
            new_items = self._count_recipe_prime_ingredients(ingredient, inventory)
            for n_item, n_quant in new_items.items():
                if n_item in items:
                    items[n_item] += quantity * n_quant
                else:
                    items[n_item] = quantity * n_quant
        return items

    def _count_creation_steps(self, search_node, item_type: str, needed: int):
        """
        Counts the number of steps still needed to create an item, disregarding mining costs (assuming you have all the
        required prime ingredients in inventory).
        """
        steps = 0
        recipe = self.initial_state.get_recipe_for(item_type)
        if recipe is None:
            return steps
        else:
            step1_ingredients = get_ingredients_for_recipe(recipe)

            for ingredient, quantity in step1_ingredients.items():
                fluent = f"['count_{ingredient.replace(':', '_')}']"
                quantity = quantity - search_node.state_vars[fluent]
                if quantity > 0:
                    if ingredient == ItemType.SACK_POLYISOPRENE_PELLETS:
                        # Extra steps for placing and collecting tree tap
                        steps += 2
                        ingredient = ItemType.TREE_TAP.value
                        fluent = f"['count_{ingredient.replace(':', '_')}']"
                        quantity = quantity - search_node.state_vars[fluent]
                        if quantity <= 0:
                            continue
                    num_out = get_outputs_of_recipe(recipe)[item_type]
                    # One step for the crafting, multipy by how many times we will need to craft it.
                    steps += 1 + self._count_creation_steps(search_node, ingredient, quantity) / num_out

        return steps * needed

    def _cost_to_create_item(self, search_node: SearchState, item_type: str):
        """
        returns the expected cost in search steps to create one item of given type, including recipe cost and ingredient
        mining cost.
        """
        h_value = 0
        inventory = self._get_inventory(search_node)
        prime_inds = self._count_recipe_prime_ingredients(item_type, inventory)
        # items to be mined from some kind of block
        for source, results in self.metamodel.break_block_to_outcome.items():
            # search through all minable block types
            item, count = results
            if item in prime_inds.keys():
                fluent = f"['count_{item.replace(':', '_')}']"
                inds_still_needed = max(0, prime_inds[item] - search_node.state_vars[fluent])
                source_cells_needed = ceil(inds_still_needed / count)
                source_cells_accbl, source_cells_tot = self._count_cells_of_type(search_node, source)
                h_value += source_cells_needed  # need to mine this many cells
                h_value += source_cells_tot - source_cells_accbl  # need to make these cells accessible
                h_value += max(0, source_cells_needed - source_cells_tot)  # cells that need to be created somehow

        creation_steps = self._count_creation_steps(search_node, item_type, 1)
        h_value += creation_steps
        return h_value

    ## Current heuristic - empirically seems to work best.
    def _only_count_once_heuristic(self, node, item_type: str, needed: int = 1):
        """
        No-so-accurate heuristic - doesn't take into account needing the same ingredients for multiple
        recipes in the tree (e.g. if you need 2 logs in two recipes\branches and you have 2 logs, the cost for logs
        will be computed as 0).
        """
        h_value = 0
        recipe = self.initial_state.get_recipe_for(item_type)
        if recipe is None:
            # assume item must be mined from some block
            for source, results in self.metamodel.break_block_to_outcome.items():
                # search through all minable block types
                item, count = results
                if item == item_type:
                    h_value += 1 if count == 0 else 1 / count
                    # More accurate count of how many block we need to mine
                    source_cells_accbl, source_cells_tot = self._count_cells_of_type(node, source)
                    # If source is inaccessible, need extra step to make accessible.
                    # If no sources exist, maybe can make some somehow?
                    if source_cells_tot == 0:
                        h_value += 1
                    elif source_cells_accbl == 0:
                        h_value += 1
            if h_value == 0:
                # No known source - should still return some value?
                h_value += 1
        else:
            step1_ingredients = get_ingredients_for_recipe(recipe)

            for ingredient, quantity in step1_ingredients.items():
                fluent = f"['count_{ingredient.replace(':', '_')}']"
                quantity = quantity - node.state_vars[fluent]
                if quantity > 0:
                    if ingredient == ItemType.SACK_POLYISOPRENE_PELLETS:
                        # Extra steps for placing and collecting tree tap
                        h_value += 2
                        ingredient = ItemType.TREE_TAP.value
                        fluent = f"['count_{ingredient.replace(':', '_')}']"
                        quantity = quantity - node.state_vars[fluent]
                        if quantity <= 0:
                            continue
                    num_out = get_outputs_of_recipe(recipe)[item_type]
                    # One step for the crafting, multipy by how many times we will need to craft it.
                    h_value += 1 + self._only_count_once_heuristic(node, ingredient, quantity) / num_out

        return h_value * needed

    def evaluate(self, node: SearchState):
        # Check if we have ingredients of pogo stick
        pogo_count = node.state_vars["['count_polycraft_wooden_pogo_stick']"]
        n_value = 0
        if pogo_count > 0:
            n_value = 0
        else:
            if node.predecessor is not None and node.predecessor.predecessor is not None \
                    and node.predecessor_action.name == node.predecessor.predecessor_action.name:
                n_value += 9999
                # Some value to prevent these state being explored unless other routes don't exist.
                # One of the situations we want to avoid is teleporting back and forth - so if the last action was
                # teleporting, don't teleport again.
            n_value += self._only_count_once_heuristic(node, ItemType.WOODEN_POGO_STICK.value)

        node.h = n_value
        return n_value


class OpenDoorHeuristic(AbstractHeuristic):
    """ Heuristic for polycraft to be used by the Nyx planner when solving an open door task"""

    def __init__(self, door_cell: str):
        super().__init__()
        self.door_cell = door_cell
        pddl_door_cell_name = PddlGameMapCellType.get_cell_object_name(door_cell)
        self._is_door_accessible_var = f"['{Predicate.door_is_accessible.name}', '{pddl_door_cell_name}']"
        self._is_open_var = f"['{Predicate.open.name}', '{pddl_door_cell_name}']"
        self._passed_door_var = f"['{Predicate.passed_door.name}', '{pddl_door_cell_name}']"

    def evaluate(self, node):
        # Some indication of remaining steps to open the door.
        if node.state_vars[self._passed_door_var]:
            return 0
        if node.state_vars[self._is_open_var]:
            return 1
        if node.state_vars[self._is_door_accessible_var]:
            return 2
        else:
            return 3  # Can improve this


class CollectFromSafeHeuristic(AbstractHeuristic):
    """ Heuristic for polycraft to be used by the Nyx planner when solving an collect from safe task"""

    def __init__(self, safe_cell: str):
        super().__init__()
        self.safe_cell = safe_cell

    def evaluate(self, node):
        return 0  # TODO: IMPROVE THIS


class MakeCellAccessibleHeuristic(AbstractHeuristic):
    """ Heuristic for polycraft to be used by the Nyx planner when solving an open door task"""

    def __init__(self, cell: str):
        super().__init__()
        self.cell = PddlGameMapCellType.get_cell_object_name(cell)
        self.pddl_cell_name = {PddlGameMapCellType.get_cell_object_name(self.cell)}
        self._is_accessible_var = f"['{Predicate.isAccessible.name}', '{self.pddl_cell_name}']"

    def _steve_distance(self, node):
        steve_pos = node.state_vars[f"['{Function.Steve_x.name}']"], 0, node.state_vars[f"['{Function.Steve_z.name}']"]
        dist = compute_cell_distance(steve_pos, cell_to_coordinates(self.cell))

    def evaluate(self, node):
        # Can improve this, but the cost of calculation might not be worthwhile (e.g. how many surrounding cells we need
        #  to dig through)
        if node.state_vars[self._is_accessible_var]:
            return 0
        else:
            return self._steve_distance(node)


########### Actions

###### PDDL ACTIONS, PROCESSES, AND EVENTS

class PddlTeleportActionGenerator(PddlPolycraftActionGenerator):
    """
    Creates actions for teleporting to any cell.
    """

    def __init__(self):
        super().__init__("teleport_to")

    def to_pddl(self, meta_model: MetaModel) -> PddlPlusWorldChange:
        pddl_action = PddlPlusWorldChange(WorldChangeTypes.action)
        pddl_action.name = self.pddl_name
        pddl_action.parameters.append(["?to", "-", "cell"])

        air_idx = meta_model.block_type_to_idx[BlockType.AIR.value]
        pddl_action.preconditions.append([Predicate.isAccessible.name, '?to'])
        pddl_action.preconditions.append(["=", [Function.cell_type.name, "?to"], f"{air_idx}"])
        pddl_action.preconditions.append(["or",
                                          ["and", ["=", [Function.cell_x.name, "?to"], [Function.Steve_x.name]],
                                           ["=", [Function.cell_z.name, "?to"], ["+", [Function.Steve_z.name], "1"]]],
                                          ["and", ["=", [Function.cell_x.name, "?to"], [Function.Steve_x.name]],
                                           ["=", [Function.cell_z.name, "?to"], ["-", [Function.Steve_z.name], "1"]]],
                                          ["and", ["=", [Function.cell_z.name, "?to"], [Function.Steve_z.name]],
                                           ["=", [Function.cell_x.name, "?to"], ["+", [Function.Steve_x.name], "1"]]],
                                          ["and", ["=", [Function.cell_z.name, "?to"], [Function.Steve_z.name]],
                                           ["=", [Function.cell_x.name, "?to"], ["-", [Function.Steve_x.name], "1"]]],
                                          ])

        pddl_action.effects.append(["assign", Function.Steve_x.to_pddl(), [Function.cell_x.name, '?to']])
        pddl_action.effects.append(["assign", Function.Steve_z.to_pddl(), [Function.cell_z.name, '?to']])
        return pddl_action

    def to_polycraft(self, parameter_binding: dict) -> PolycraftAction:
        cell = parameter_binding["?to"]
        return PolyTP(cell, dist=1)


class PddlPlaceTreeTapActionGenerator(PddlPolycraftActionGenerator):
    """ An action corresponding to placing a tree tap
    New PDDL:
            (:action place_tree_tap
            :precondition (and
                (isAccessible HARD_CODED_AIR)
                (isAccessible HARD_CODED_TREE)
                (cell_type HARD_CODED_AIR {BlockType.AIR.value})
                (cell_type HARD_CODED_TREE {BlockType.LOG.value})
                (>= (count_{ItemType.TREE_TAP.name}) 1)
            )
            :effect (and
                (decrease (count_{ItemType.TREE_TAP}) 1)
                (assign (cell_type HARD_CODED_AIR) {ItemType.TREE_TAP.value})
            )
        )
    Old PDDL:
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
    """

    def __init__(self, tree_cell: str, air_cell: str):
        super().__init__(f"place_tree_tap_{tree_cell}_{air_cell}")
        self.tree = PddlGameMapCellType.get_cell_object_name(tree_cell)
        self.tap_cell = PddlGameMapCellType.get_cell_object_name(air_cell)
        self.tap_cell_binding = air_cell

    def to_pddl(self, meta_model: PolycraftMetaModel) -> PddlPlusWorldChange:
        pddl_action = PddlPlusWorldChange(WorldChangeTypes.action)
        pddl_action.name = self.pddl_name
        pddl_action.preconditions.append([Predicate.isAccessible.name, self.tree])
        log_idx = meta_model.block_type_to_idx[BlockType.LOG.value]
        pddl_action.preconditions.append(["=", [Function.cell_type.name, self.tree], f"{log_idx}"])
        pddl_action.preconditions.append([Predicate.isAccessible.name, self.tap_cell])
        air_idx = meta_model.block_type_to_idx[BlockType.AIR.value]
        pddl_action.preconditions.append(["=", [Function.cell_type.name, self.tap_cell], f"{air_idx}"])
        pddl_action.preconditions.append([">=", [f"count_{ItemType.TREE_TAP.value}", ], "1"])

        pddl_action.effects.append(["decrease", [f"count_{ItemType.TREE_TAP.value}", ], "1"])
        tree_tap_idx = meta_model.block_type_to_idx[BlockType.TREE_TAP.value]
        pddl_action.effects.append(["assign", [Function.cell_type.name, self.tap_cell], f"{tree_tap_idx}"])

        return pddl_action

    def to_polycraft(self, parameter_binding: dict) -> PolycraftAction:
        return TeleportToAndPlaceTreeTap(self.tap_cell_binding)


class PddlCollectActionGenerator(PddlPolycraftActionGenerator):
    """ An action corresponding to collecting an item from a cell (COLLECT in the Polycraft API)
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
    """

    def __init__(self, collect_from_block_type: str, item_type_to_collect: str, quantity):
        super().__init__(f"collect_{item_type_to_collect}_from_{collect_from_block_type}")
        self.item_type_to_collect = item_type_to_collect
        self.collect_from_block_type = collect_from_block_type
        self.quantity = quantity

    def to_pddl(self, meta_model: PolycraftMetaModel) -> PddlPlusWorldChange:
        pddl_action = PddlPlusWorldChange(WorldChangeTypes.action)
        pddl_action.name = self.pddl_name
        pddl_action.parameters.append(["?c", "-", "cell"])

        pddl_action.preconditions.append([Predicate.isAccessible.name, "?c"])
        cell_type_idx = meta_model.block_type_to_idx[self.collect_from_block_type]
        pddl_action.preconditions.append(["=", [Function.cell_type.name, "?c"], f"{cell_type_idx}"])

        pddl_action.effects.append(["increase", [f"count_{self.item_type_to_collect}", ], str(self.quantity)])
        pddl_action.effects.append(["assign", Function.Steve_x.to_pddl(), [Function.cell_x.name, '?c']])
        pddl_action.effects.append(["assign", Function.Steve_z.to_pddl(), [Function.cell_z.name, '?c']])
        return pddl_action

    def to_polycraft(self, parameter_binding: dict) -> PolycraftAction:
        return TeleportToAndCollect(parameter_binding['?c'])


class PddlCollectFromTreeTapActionGenerator(PddlPolycraftActionGenerator):
    """ An action corresponding to collecting an item from a cell (COLLECT in the Polycraft API)
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
    """

    def __init__(self):
        super().__init__(f"collect_from_tree_tap")

    def to_pddl(self, meta_model: PolycraftMetaModel) -> PddlPlusWorldChange:
        pddl_action = PddlPlusWorldChange(WorldChangeTypes.action)
        pddl_action.name = self.pddl_name
        pddl_action.parameters.append(["?c", "-", "cell"])

        tree_tap_idx = meta_model.block_type_to_idx[BlockType.TREE_TAP.value]
        pddl_action.preconditions.append(["=", [Function.cell_type.name, "?c"], f"{tree_tap_idx}"])

        sack_of_pellets_per_collect = meta_model.collect_block_to_outcome[BlockType.TREE_TAP.value][1]
        pddl_action.effects.append(["increase", [f"count_{ItemType.SACK_POLYISOPRENE_PELLETS.value}", ],
                                    str(sack_of_pellets_per_collect)])
        return pddl_action

    def to_polycraft(self, parameter_binding: dict) -> PolycraftAction:
        return TeleportToAndCollect(parameter_binding['?c'])


class PddlBreakActionGenerator(PddlPolycraftActionGenerator):
    """ An action for moving to a cell, breaking it, and collecting the resulting item.

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
    """

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

        pddl_action.preconditions.append([Predicate.isAccessible.name, "?c"])
        cell_type_idx = meta_model.block_type_to_idx[self.block_type_to_break]
        pddl_action.preconditions.append(["=", [Function.cell_type.name, "?c"], f"{cell_type_idx}"])
        if self.needs_iron_pickaxe:
            iron_pickaxe_idx = meta_model.item_type_to_idx[ItemType.IRON_PICKAXE.value]
            pddl_action.preconditions.append(["=", ["selectedItem", ], f"{iron_pickaxe_idx}"])


        pddl_action.effects.append(["increase", [f"count_{self.item_type_to_collect}", ], str(self.items_per_block)])
        air_cell_idx = meta_model.block_type_to_idx[BlockType.AIR.value]
        pddl_action.effects.append(["assign", ["cell_type", "?c"], f"{air_cell_idx}"])
        pddl_action.effects.append(["assign", Function.Steve_x.to_pddl(), [Function.cell_x.name, '?c']])
        pddl_action.effects.append(["assign", Function.Steve_z.to_pddl(), [Function.cell_z.name, '?c']])
        return pddl_action

    def to_polycraft(self, parameter_binding: dict) -> PolycraftAction:
        return TeleportToBreakAndCollect(parameter_binding['?c'])


class PddlSelectActionGenerator(PddlPolycraftActionGenerator):
    """ An action for selecting an item from the inventory
        ; SELECT
        (:action select_{item_type.name}
            :precondition (and
                (>= (count_{item_type.name}) 1)
            )
            :effect (and
                (assign (selectedItem) {item_type.value})
            )
        )
    """

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
    """ An action for using a dor
        ; Use door (USE)
        (:action select_{item_type.name}
            :precondition (and
                (>= (count_{item_type.name}) 1)
            )
            :effect (and
                (assign (selectedItem) {item_type.value})
            )
        )
    """

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
    """ An action generator for opening a safe and collecting it """

    def __init__(self):
        super().__init__(f"open_safe_and_collect")

    def to_pddl(self, meta_model: MetaModel) -> PddlPlusWorldChange:
        pddl_action = PddlPlusWorldChange(WorldChangeTypes.action)
        pddl_action.name = self.pddl_name
        pddl_action.parameters.append(["?c", "-", PddlType.safe_cell.name])

        pddl_action.preconditions.append([Predicate.safe_is_accessible.name, "?c"])
        pddl_action.preconditions.append([">=", [f"count_{ItemType.KEY.value}"], "1"])


        pddl_action.effects.append([Predicate.safe_collected.name, "?c"])

        return pddl_action

    def to_polycraft(self, parameter_binding: dict) -> PolycraftAction:
        return OpenSafeAndCollect(parameter_binding["?c"])


class PddlTradeActionGenerator(PddlPolycraftActionGenerator):
    """ An action for trading an item with a trader
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
    """

    def __init__(self, trader_id: str, trade_idx: int, trade):
        super().__init__(f"trade_recipe_{trader_id}_{trade_idx}")
        self.trader_id = trader_id
        self.trade_idx = trade_idx
        self.trade = trade

    def to_pddl(self, meta_model: MetaModel) -> PddlPlusWorldChange:
        pddl_action = PddlPlusWorldChange(WorldChangeTypes.action)
        pddl_action.name = self.pddl_name
        pddl_action.parameters.append(["?trader_loc", "-", "cell"])

        pddl_action.preconditions.append([Predicate.isAccessible.name, "?trader_loc"])

        for input in self.trade["inputs"]:
            item_type = input['Item']
            quantity = input['stackSize']
            pddl_action.preconditions.append([">=", [f"count_{item_type}"], str(quantity)])
            pddl_action.effects.append(["decrease", [f"count_{item_type}"], str(quantity)])

        pddl_action.preconditions.append([f"trader_{self.trader_id}_at", "?trader_loc"])

        for output in self.trade["outputs"]:
            item_type = output['Item']
            quantity = output['stackSize']
            pddl_action.effects.append(["increase", [f"count_{item_type}"], str(quantity)])

        return pddl_action

    def to_polycraft(self, parameter_binding: dict) -> PolycraftAction:
        return TeleportToTraderAndTrade(self.trader_id, self.trade)


class PddlPolycraftEvent:
    """ A class representing a PDDL+ action in polycraft """

    def to_pddl(self, meta_model: PolycraftMetaModel) -> PddlPlusWorldChange:
        """ This method should be implemented by sublcasses and output a string representation of the corresponding PDDL+ action """
        raise NotImplementedError()


class CellAccessibleEvent(PddlPolycraftEvent):
    """
    Marks a cell as newly accessible. General form:
    paramters: cell
    preconditions: not accessible
                   next to Steve
    effects: accessible
    """

    def to_pddl(self, meta_model: PolycraftMetaModel):
        pddl_event = PddlPlusWorldChange(WorldChangeTypes.event)
        pddl_event.name = "cell_accessible"
        pddl_event.parameters.append(["?c", "-", PddlType.cell.name])
        pddl_event.preconditions.append(["not", [Predicate.isAccessible.name, "?c"]])
        pddl_event.preconditions.append(["or",
                                         ["and", ["=", [Function.cell_x.name, "?c"], [Function.Steve_x.name]],
                                          ["=", [Function.cell_z.name, "?c"], ["+", [Function.Steve_z.name], "1"]]],
                                         ["and", ["=", [Function.cell_x.name, "?c"], [Function.Steve_x.name]],
                                          ["=", [Function.cell_z.name, "?c"], ["-", [Function.Steve_z.name], "1"]]],
                                         ["and", ["=", [Function.cell_z.name, "?c"], [Function.Steve_z.name]],
                                          ["=", [Function.cell_x.name, "?c"], ["+", [Function.Steve_x.name], "1"]]],
                                         ["and", ["=", [Function.cell_z.name, "?c"], [Function.Steve_z.name]],
                                          ["=", [Function.cell_x.name, "?c"], ["-", [Function.Steve_x.name], "1"]]],
                                         ])

        pddl_event.effects.append([Predicate.isAccessible.name, "?c"])
        return pddl_event


class DoorAccessibleEvent(PddlPolycraftEvent):
    """
    Make a door accessible
    """
    def to_pddl(self, meta_model: PolycraftMetaModel):
        pddl_event = PddlPlusWorldChange(WorldChangeTypes.event)
        pddl_event.name = "door_accessible"
        pddl_event.parameters.append(["?c", "-", PddlType.door_cell.name])

        pddl_event.preconditions.append(["not", [Predicate.door_is_accessible.name, "?c"]])
        air_idx = meta_model.block_type_to_idx[BlockType.AIR.value]
        pddl_event.preconditions.append(["or",
                                         ["and", ["=", [Function.cell_x.name, "?c"], [Function.Steve_x.name]],
                                          ["=", [Function.cell_z.name, "?c"], ["+", [Function.Steve_z.name], "1"]]],
                                         ["and", ["=", [Function.cell_x.name, "?c"], [Function.Steve_x.name]],
                                          ["=", [Function.cell_z.name, "?c"], ["-", [Function.Steve_z.name], "1"]]],
                                         ["and", ["=", [Function.cell_z.name, "?c"], [Function.Steve_z.name]],
                                          ["=", [Function.cell_x.name, "?c"], ["+", [Function.Steve_x.name], "1"]]],
                                         ["and", ["=", [Function.cell_z.name, "?c"], [Function.Steve_z.name]],
                                          ["=", [Function.cell_x.name, "?c"], ["-", [Function.Steve_x.name], "1"]]],
                                         ])

        pddl_event.effects.append([Predicate.door_is_accessible.name, "?c"])
        return pddl_event


class SafeAccessibleEvent(PddlPolycraftEvent):
    """
    Make a safe accessible
    """
    def to_pddl(self, meta_model: PolycraftMetaModel):
        pddl_event = PddlPlusWorldChange(WorldChangeTypes.event)
        pddl_event.name = "safe_accessible"
        pddl_event.parameters.append(["?c", "-", PddlType.safe_cell.name])
        pddl_event.preconditions.append(["not", [Predicate.safe_is_accessible.name, "?c"]])
        pddl_event.preconditions.append(["or",
                                         ["and", ["=", [Function.cell_x.name, "?c"], [Function.Steve_x.name]],
                                          ["=", [Function.cell_z.name, "?c"], ["+", [Function.Steve_z.name], "1"]]],
                                         ["and", ["=", [Function.cell_x.name, "?c"], [Function.Steve_x.name]],
                                          ["=", [Function.cell_z.name, "?c"], ["-", [Function.Steve_z.name], "1"]]],
                                         ["and", ["=", [Function.cell_z.name, "?c"], [Function.Steve_z.name]],
                                          ["=", [Function.cell_x.name, "?c"], ["+", [Function.Steve_x.name], "1"]]],
                                         ["and", ["=", [Function.cell_z.name, "?c"], [Function.Steve_z.name]],
                                          ["=", [Function.cell_x.name, "?c"], ["-", [Function.Steve_x.name], "1"]]],
                                         ])
        pddl_event.effects.append([Predicate.safe_is_accessible.name, "?c"])
        return pddl_event


class PddlCraftActionGenerator(PddlPolycraftActionGenerator):
    """ An action for crafting an item
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
    """

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

        if self.needs_crafting_table:  # TODO: Not robust to problems with multiple crafting tables Yoni: Isn't it?
            pddl_action.parameters.append(["?from", "-", "cell"])

            pddl_action.preconditions.append([Predicate.isAccessible.name, "?from"])
            crafting_table_idx = meta_model.block_type_to_idx[BlockType.CRAFTING_TABLE.value]
            pddl_action.preconditions.append(["=", ["cell_type", "?from", ], f"{crafting_table_idx}"])

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


class PddlMoveThroughDoorActionGenerator(PddlPolycraftActionGenerator):
    """ An action for moving through a door
        ; MOVE w w
        (:action move_through_door_{door_cell}
            :parameters (?cell - door_cell)
            :precondition( and
                (isAccessible ?cell)
                (open ?cell)
            )
        )
    """

    def __init__(self):
        super().__init__("move_through_door")

    def to_pddl(self, meta_model: PolycraftMetaModel) -> PddlPlusWorldChange:
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
    """ The types of tasks this meta model object supports """
    CRAFT_POGO = CreatePogoTask
    EXPLORE_DOOR = ExploreDoorTask
    MAKE_CELL_ACCESSIBLE = MakeCellAccessibleTask
    COLLECT_FROM_SAFE = CollectFromSafeTask

    def create_instance(self):
        """ Create an instance of this task"""
        return self.value.__new__(self.value)
