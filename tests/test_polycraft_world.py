import pickle
import pytest
import settings
import logging
import pathlib

from utils.polycraft_utils import *
from agent.planning.polycraft_planning.actions import *
import worlds.polycraft_world as poly
from agent.polycraft_hydra_agent import PolycraftObservation, PolycraftManualAgent

logging.basicConfig(format='%(name)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TestPolycraftWorld")

TEST_LEVEL = pathlib.Path(
    settings.POLYCRAFT_NON_NOVELTY_LEVEL_DIR) / "POGO_L00_T01_S01_X0100_U9999_V0_G00000_I0020_N0.json"

@pytest.fixture(scope="module")
def launch_polycraft():
    logger.info("starting")

    env = poly.Polycraft(launch=True)
    yield env
    logger.info("teardown tests")
    env.kill()

def _setup_env(env):
    ''' Sets up the environment and agent, and return the current state '''
    agent = PolycraftManualAgent()
    agent.commands.clear()
    env.init_selected_level(TEST_LEVEL)
    agent.start_level(env)  # Collect trades and recipes
    state = env.get_current_state()
    return agent, state


@pytest.mark.parametrize('execution_number', range(5))
def test_agent_break_and_collect(launch_polycraft: poly.Polycraft, execution_number):
    ''' Test going to an log block and mining it '''
    env = launch_polycraft
    agent, state = _setup_env(env)

    log_cells = state.get_cells_of_type(poly.ItemType.LOG.value, only_accessible=True)
    assert (len(log_cells) > 0)

    cell = log_cells[execution_number % len(log_cells)]  # Change the log cell we choose
    agent.commands.append(PolyBreakAndCollect(cell))
    action = agent.choose_action(state)
    after_state, step_cost = agent.do(action, env)

    if action.success:
        diff = after_state.diff(state)
        assert (has_new_item(diff))
        assert (after_state.has_item(poly.ItemType.LOG.value))
    else: # Failure may be due to other pogo-ist. Assert that this is possible
        dist_to_pogoist = distance_to_nearest_pogoist(after_state, cell)
        assert(dist_to_pogoist<=2)

@pytest.mark.parametrize('execution_number', range(5))
def test_mining_two_logs(launch_polycraft: poly.Polycraft, execution_number):
    ''' Test mining two oak blocks and collecting the resulting logs '''
    env = launch_polycraft
    agent, state = _setup_env(env)

    log_cells = state.get_cells_of_type(poly.ItemType.LOG.value, only_accessible=True)
    assert(len(log_cells)>1)
    cell1 = log_cells[execution_number % len(log_cells)]
    cell2 = log_cells[(execution_number +1) % len(log_cells)]

    action = PolyBreakAndCollect(cell1)
    after_state, step_cost = agent.do(action, env)
    assert(has_new_item(after_state.diff(state)))
    assert(after_state.has_item(poly.ItemType.LOG.value))
    state = after_state

    if cell2 not in state.get_cells_of_type(poly.ItemType.LOG.value, only_accessible=True):
        logger.info("Log cell disappeared, probably the other pogoist took it")
        return

    action = PolyBreakAndCollect(cell2)
    after_state, step_cost = agent.do(action, env)
    if action.success:
        assert(has_new_item(after_state.diff(state)))
        log_entries = after_state.get_inventory_entries_of_type(poly.ItemType.LOG.value)
        assert(len(log_entries)==1)
        assert(after_state.inventory[log_entries[0]]['count']==2)
    else:
        dist_to_pogoist = distance_to_nearest_pogoist(after_state, cell2)
        assert(dist_to_pogoist<=2)

@pytest.mark.parametrize('execution_number', range(5))
def test_select_iron_pickaxe(launch_polycraft, execution_number):
    ''' Test selecting an iron pickaxe from the inventory '''
    env = launch_polycraft
    agent, state = _setup_env(env)

    # Assert no item has been selected
    selected_item = state.get_selected_item()
    assert(selected_item is None or len(selected_item)==0) # No item selected initially

    # Assert we have an iron pickaxe
    iron_pickaxe_entries = state.get_inventory_entries_of_type(poly.ItemType.IRON_PICKAXE.value)
    assert(len(iron_pickaxe_entries)>0) # We have an iron pickaxe in our inventory

    # Select the iron pickaxe
    action = poly.PolySelectItem(poly.ItemType.IRON_PICKAXE.value)
    after_state, step_cost = agent.do(action, env)

    assert(after_state.get_selected_item()==poly.ItemType.IRON_PICKAXE.value)


@pytest.mark.parametrize('execution_number', range(5))
def test_agent_mine_with_pickaxe(launch_polycraft: poly.Polycraft, execution_number:int):
    ''' Select a pickaxe and use it'''
    env = launch_polycraft
    agent, state = _setup_env(env)

    # Select pickaxe
    iron_pickaxe_entries = state.get_inventory_entries_of_type(poly.ItemType.IRON_PICKAXE.value)
    assert(len(iron_pickaxe_entries) > 0)
    assert(state.get_selected_item()!=poly.ItemType.IRON_PICKAXE.value)

    action = poly.PolySelectItem(poly.ItemType.IRON_PICKAXE.value)
    after_state, step_cost = agent.do(action, env)
    assert(action.success)
    assert(after_state.get_selected_item()==poly.ItemType.IRON_PICKAXE.value)
    state = after_state

    # Mine a log (with the pickaxe)
    log_cells = state.get_cells_of_type(poly.ItemType.LOG.value)
    assert (len(log_cells) > 0)
    accessible_log_cells = []
    for cell in log_cells:
        if state.game_map[cell]["isAccessible"]:
            accessible_log_cells.append(cell)

    assert(len(accessible_log_cells)>0)
    cell = accessible_log_cells[execution_number % len(accessible_log_cells)]  # Change the log cell we choose
    action = PolyBreakAndCollect(cell)
    after_state, step_cost = agent.do(action, env)
    if action.success:
        assert (after_state.has_item(poly.ItemType.LOG.value))
    else: # Failure may be due to other pogo-ist. Assert that this is possible
        dist_to_pogoist = distance_to_nearest_pogoist(after_state, cell)
        assert(dist_to_pogoist<=2)


@pytest.mark.parametrize('execution_number', range(1))
def test_craft_items(launch_polycraft: poly.Polycraft, execution_number):
    ''' Test mining logs, crafting planks from the logs, and crafting sticks from planks '''
    env = launch_polycraft
    agent, state = _setup_env(env)

    # Mine 2 logs
    state = _mine_logs(agent, env, state)
    state = _mine_logs(agent, env, state)

    # Make planks
    planks_recipe, state = _craft_planks(agent, env, state)

    # Make stick
    state = _craft_stick(agent, env, state)

    # Make more planks
    planks_recipe, state = _craft_planks(agent, env, state)

    # Move to a crafting table
    after_state, step_cost = _go_to_crafting_table(agent, env, state)
    state = after_state

    # Craft
    after_state = _craft_tree_tap(agent, env, state)
    state = after_state

    # Find a place for placing the tree tap
    log_cells = state.get_cells_of_type(poly.BlockType.LOG.value, only_accessible=True)
    assert (len(log_cells) > 0)
    cells_to_place_tree_tap = []
    for cell in log_cells:
        for adjacent_cell in get_adjacent_cells(cell):
            if adjacent_cell in state.game_map and \
                    state.game_map[adjacent_cell]["name"]==BlockType.AIR.value and \
                    state.game_map[adjacent_cell]["isAccessible"]:
                cells_to_place_tree_tap.append(adjacent_cell)
    assert(len(cells_to_place_tree_tap)>0)
    cell = cells_to_place_tree_tap[0]
    after_state, step_cost = agent.do(TeleportAndFaceCell(cell), env)
    assert(agent.current_observation.actions[-1].success)
    state = after_state

    # Place the tree tap
    after_state, step_cost = agent.do(PolyPlaceTreeTap(), env)
    assert(agent.current_observation.actions[-1].success)
    assert(after_state.facing_block['name']==ItemType.TREE_TAP.value) # Assert facing the tree tap
    state = after_state

    # Collect the item
    after_state, step_cost = agent.do(PolyCollect(), env)
    assert(agent.current_observation.actions[-1].success)
    assert(after_state.count_items_of_type(ItemType.SACK_POLYISOPRENE_PELLETS.value)>0)


def _craft_tree_tap(agent, env, state):
    ''' Craft a tree tap'''
    tree_tap_recipes = state.get_all_recipes_for(poly.ItemType.TREE_TAP.value)
    assert (len(tree_tap_recipes) == 1)
    recipe = tree_tap_recipes[0]
    missing_ingridients = compute_missing_ingridients(recipe, state)
    assert (len(missing_ingridients) == 0)
    action = PolyCraftItem.create_action(recipe)
    after_state, step_cost = agent.do(action, env)
    assert (after_state.has_item(poly.ItemType.TREE_TAP.value))
    return after_state


def _craft_stick(agent, env, state):
    ''' Crafts a stick '''
    stick_recipes = state.get_all_recipes_for(poly.ItemType.STICK.value)
    assert (len(stick_recipes) == 1)
    recipe = stick_recipes[0]
    action = PolyCraftItem.create_action(recipe)
    after_state, step_cost = agent.do(action, env)
    assert (after_state.has_item(poly.ItemType.STICK.value))
    state = after_state
    return state


def _craft_planks(agent, env, state):
    ''' Crafts planks'''
    planks_recipes = state.get_all_recipes_for(poly.ItemType.PLANKS.value)
    assert (len(planks_recipes) == 1)
    planks_recipe = planks_recipes[0]
    action = PolyCraftItem.create_action(planks_recipe)
    after_state, step_cost = agent.do(action, env)
    assert (after_state.has_item(poly.ItemType.PLANKS.value))
    state = after_state
    return planks_recipe, state


def _mine_logs(agent, env, state):
    ''' Go to a log block cell, mine it, and collect it'''
    log_cells = state.get_cells_of_type(poly.ItemType.LOG.value)
    assert (len(log_cells) > 0)
    action = PolyBreakAndCollect(log_cells[0])
    after_state, step_cost = agent.do(action, env)
    assert (after_state.has_item(poly.ItemType.LOG.value))
    assert (has_new_item(after_state.diff(state)))
    state = after_state
    return state


@pytest.mark.parametrize('execution_number', range(5))
def test_mine_diamonds(launch_polycraft: poly.Polycraft, execution_number):
    ''' Test mining logs, crafting planks from the logs, and crafting sticks from planks '''
    env = launch_polycraft
    agent, state = _setup_env(env)

    # Select iron pickaxe
    action = poly.PolySelectItem(poly.ItemType.IRON_PICKAXE.value)
    after_state, step_cost = agent.do(action, env)
    assert(action.success)
    assert(after_state.get_selected_item()==poly.ItemType.IRON_PICKAXE.value)
    state = after_state

    # Mine
    diamond_cells = state.get_cells_of_type(poly.BlockType.DIAMOND_ORE.value, only_accessible=True)
    mined_diamonds = 0
    current_diamonds_in_inventory = len(state.get_inventory_entries_of_type(poly.ItemType.DIAMOND.value))
    assert(current_diamonds_in_inventory==0)
    while current_diamonds_in_inventory<9 and len(diamond_cells)>0:
        # Mine the diamond
        cell = diamond_cells[0]
        action = PolyBreakAndCollect(cell)
        after_state, step_cost = agent.do(action, env)

        if action.success==True:
            assert(has_new_item(after_state.diff(state)))
            old_diamonds_in_inventory = current_diamonds_in_inventory
            current_diamonds_in_inventory = after_state.count_items_of_type(poly.ItemType.DIAMOND.value)
            assert (current_diamonds_in_inventory >= old_diamonds_in_inventory)
            logger.info(f'Diamond collected from {cell}!')
        else:
            # Diamond not mined, maybe pogoist stole it
            dist_to_pogoist = distance_to_nearest_pogoist(after_state, cell)
            assert(dist_to_pogoist <= 2)
            logger.info(f'Diamond mined by pogoist stole it from {cell} :(')

        # Find next diamond to mine
        state = after_state
        diamond_cells = state.get_cells_of_type(poly.BlockType.DIAMOND_ORE.value, only_accessible=True)

    logger.info(f'A total of {current_diamonds_in_inventory} were mined!')
    assert(current_diamonds_in_inventory>=9)

    # Go to crafting table
    after_state, step_cost = _go_to_crafting_table(agent, env, state)
    state = after_state

    # Craft the diamond block
    diamond_block_recipes = state.get_all_recipes_for(poly.ItemType.DIAMOND_BLOCK.value)
    assert (len(diamond_block_recipes) == 1)
    recipe = diamond_block_recipes[0]
    action = PolyCraftItem.create_action(recipe)
    after_state, step_cost = agent.do(action, env)
    assert (after_state.has_item(poly.ItemType.DIAMOND_BLOCK.value))


def _go_to_crafting_table(agent, env, state):
    ''' Teleport to an accessible crafting table'''
    crafting_table_cells = state.get_cells_of_type(poly.BlockType.CRAFTING_TABLE.value, only_accessible=True)
    assert (len(crafting_table_cells) == 1)
    target_cell = crafting_table_cells[0]
    action = poly.PolyTP(target_cell, dist=1)
    after_state, step_cost = agent.do(action, env)
    assert (action.success)
    distance_to_table = compute_cell_distance(cell_to_coordinates(target_cell), after_state.location['pos'])
    assert (distance_to_table < 2)
    return after_state, step_cost


def test_trade_logs(launch_polycraft: poly.Polycraft):
    ''' Test getting logs and trading them for titanium '''
    env = launch_polycraft
    agent, state = _setup_env(env)

    # Select iron pickaxe
    action = poly.PolySelectItem(poly.ItemType.IRON_PICKAXE.value)
    after_state, step_cost = agent.do(action, env)
    assert(action.success)
    assert(after_state.get_selected_item()==poly.ItemType.IRON_PICKAXE.value)
    state = after_state

    # Collect block of platinum
    action = CollectAndMineItem(BlockType.BLOCK_OF_PLATINUM.value, 1, [BlockType.BLOCK_OF_PLATINUM.value])
    after_state, step_cost = agent.do(action, env)
    assert(action.success)
    assert(after_state.count_items_of_type(BlockType.BLOCK_OF_PLATINUM.value)>=1)
    state = after_state

    # Trade
    # Find relevant trader and trade
    trades = get_trades_for(state, desired_inputs=[{'Item': BlockType.BLOCK_OF_PLATINUM.value, 'stackSize': 1, 'slot': 0}],
                   desired_outputs=[{'Item': ItemType.BLOCK_OF_TITANIUM.value, 'stackSize': 1, 'slot': 5}])

    assert(len(trades)>0)
    (trader_id, trade) = trades[0]
    action = PolyEntityTP(trader_id,1)
    agent.do(action, env)
    assert (action.success)

    action = PolyTradeItems.create_action(trader_id, trade)
    after_state, step_cost = agent.do(action, env)
    assert (action.success)
    assert(after_state.count_items_of_type(ItemType.BLOCK_OF_TITANIUM.value)>=1)
    return # All done!


def test_trade_two_titanium_blocks(launch_polycraft: poly.Polycraft):
    ''' Mine 2 platinum blocks and then trade them for two titanium blocks '''
    env = launch_polycraft
    agent, state = _setup_env(env)

    # Select iron pickaxe
    action = poly.PolySelectItem(poly.ItemType.IRON_PICKAXE.value)
    after_state, step_cost = agent.do(action, env)
    assert(action.success)
    assert(after_state.get_selected_item()==poly.ItemType.IRON_PICKAXE.value)
    state = after_state

    # Collect block of platinum
    action = CollectAndMineItem(BlockType.BLOCK_OF_PLATINUM.value, 2, [BlockType.BLOCK_OF_PLATINUM.value])
    after_state, step_cost = agent.do(action, env)
    assert(action.success)
    assert(after_state.count_items_of_type(BlockType.BLOCK_OF_PLATINUM.value)>=2)
    state = after_state

    # Find relevant trader and trade
    trades = get_trades_for(state, desired_inputs=[{'Item': BlockType.BLOCK_OF_PLATINUM.value, 'stackSize': 1, 'slot': 0}],
                   desired_outputs=[{'Item': ItemType.BLOCK_OF_TITANIUM.value, 'stackSize': 1, 'slot': 5}])

    assert(len(trades)>0)
    (trader_id, trade) = trades[0]
    action = PolyEntityTP(trader_id,1)
    agent.do(action, env)
    assert (action.success)

    # Trade 1
    action = PolyTradeItems(trader_id, [{"Item": BlockType.BLOCK_OF_PLATINUM.value, "stackSize":1}])
    after_state, step_cost = agent.do(action, env)
    assert (action.success)
    assert (after_state.count_items_of_type(ItemType.BLOCK_OF_TITANIUM.value) >= 1)

    # Trade 2
    action = PolyTradeItems(trader_id, [{"Item": BlockType.BLOCK_OF_PLATINUM.value, "stackSize":1}])
    after_state, step_cost = agent.do(action, env)
    assert (action.success)
    assert (after_state.count_items_of_type(ItemType.BLOCK_OF_TITANIUM.value) >= 2)


@pytest.mark.parametrize('execution_number', range(3))
def test_open_chest(launch_polycraft: poly.Polycraft, execution_number):
    env = launch_polycraft
    agent, state = _setup_env(env)

    # Find door
    chests = state.get_cells_of_type(BlockType.PLASTIC_CHEST.value)
    for chest in chests:
        logger.info(f"Open chest at {chest}")
        agent.do(TeleportAndFaceCell(chest), env)
        assert (agent.current_observation.actions[-1].success == True)

        logger.info("Collect from the chest")
        after_state, step_cost = agent.do(PolyCollect(), env)
        assert (agent.current_observation.actions[-1].success == True)
        assert(after_state.count_items_of_type(ItemType.KEY.value)>0) # Assert found a key
        state = after_state

        door_cells = state.get_cells_of_type(BlockType.WOODER_DOOR.value, only_accessible=True)
        assert(len(door_cells)>0)
        door = door_cells[execution_number % len(door_cells)]
        after_state, step_cost = agent.do(TeleportAndFaceCell(door), env)
        assert (agent.current_observation.actions[-1].success == True)
        state = after_state

        assert(state.game_map[door]["open"] == 'false')
        agent.do(PolyUseItem(ItemType.KEY.value), env)
        assert(agent.current_observation.actions[-1].success==True)
        state = env.get_current_state()
        assert(state.game_map[door]["open"]=='true')


def test_helper_functions():
    ''' Test some of the helper functions '''
    test_path = pathlib.Path(settings.ROOT_PATH) / "tests"
    state = None
    with open(test_path / "polycraft_obs_for_test.p", "rb") as in_file:
        obs = pickle.load(in_file)
    assert(obs is not None)
    state = obs.states[1]

    item_to_recipe = get_recipe_forest(state)
    assert (poly.ItemType.WOODEN_POGO_STICK.value in item_to_recipe)
    print(" ")
    print_recipe_forest(item_to_recipe)

    item_to_trades = get_item_to_trades(state)
    print(" ")
    print_item_to_trades(item_to_trades)

    print(" ")
    type_to_cells = state.get_type_to_cells()
    for type, cells in type_to_cells.items():
        if type not in [poly.BlockType.AIR.value, poly.BlockType.BEDROCK.value]:
            print(f'{type} in cells {cells}')

    log_cells = state.get_cells_of_type(poly.ItemType.LOG.value)
    assert(len(log_cells)>0)
    for cell in log_cells:
        cell_type = state.game_map[cell]["name"]
        assert(cell_type==poly.ItemType.LOG.value)
        print(f'cell {cell} is of type {cell_type}')




# def test_other():
#     # TODO: DELETE ME
#     env = launch_polycraft
#     agent, state = _setup_env(env)
#     titanium_cells = state.get_cells_of_type(poly.ItemType.BLOCK_OF_TITANIUM.value)
#     sack_polyisoprene_pellets_cells = state.get_cells_of_type(poly.ItemType.SACK_POLYISOPRENE_PELLETS.value)
#     diamond_cells = state.get_cells_of_type(poly.ItemType.DIAMOND.value)
#     diamond_block_cells = state.get_cells_of_type(poly.ItemType.DIAMOND_BLOCK.value)