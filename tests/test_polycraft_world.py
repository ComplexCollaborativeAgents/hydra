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

    # Mine
    log_cells = state.get_cells_of_type(poly.ItemType.LOG.value)
    assert (len(log_cells) > 0)
    action = PolyBreakAndCollect(log_cells[0])
    after_state, step_cost = agent.do(action, env)
    assert(after_state.has_item(poly.ItemType.LOG.value))
    state = after_state

    # Make planks
    planks_recipe_indices = state.get_recipe_indices_for(poly.ItemType.PLANKS.value)
    assert(len(planks_recipe_indices)==1)
    recipe = state.recipes[planks_recipe_indices[0]]
    action = PolyCraftItem.create_action(recipe)
    after_state, step_cost = agent.do(action, env)
    assert(after_state.has_item(poly.ItemType.PLANKS.value))
    state = after_state

    # Make stick
    stick_recipe_indices = state.get_recipe_indices_for(poly.ItemType.STICK.value)
    assert(len(stick_recipe_indices)==1)
    recipe = state.recipes[stick_recipe_indices[0]]
    action = PolyCraftItem.create_action(recipe)
    after_state, step_cost = agent.do(action, env)
    assert(after_state.has_item(poly.ItemType.STICK.value))

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
    crafting_table_cells = state.get_cells_of_type(poly.BlockType.CRAFTING_TABLE.value, only_accessible=True)
    assert(len(crafting_table_cells)==1)
    target_cell = crafting_table_cells[0]
    action = poly.PolyTP(target_cell, dist=1)
    after_state, step_cost = agent.do(action, env)
    assert(action.success)
    distance_to_table = compute_cell_distance(cell_to_coordinates(target_cell), after_state.location['pos'])
    assert(distance_to_table<2)

    # Craft the diamond block
    diamond_block_recipe_indices = state.get_recipe_indices_for(poly.ItemType.DIAMOND_BLOCK.value)
    assert (len(diamond_block_recipe_indices) == 1)
    recipe = state.recipes[diamond_block_recipe_indices[0]]
    action = PolyCraftItem.create_action(recipe)
    after_state, step_cost = agent.do(action, env)
    assert (after_state.has_item(poly.ItemType.DIAMOND_BLOCK.value))

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

    action = PolyTradeItems(trader_id, [{"Item": BlockType.BLOCK_OF_PLATINUM.value, "stackSize":1}])
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


@pytest.mark.parametrize('execution_number', range(5))
def test_open_door(launch_polycraft: poly.Polycraft, execution_number):
    env = launch_polycraft
    agent, state = _setup_env(env)

    # Find door
    doors = state.get_cells_of_type(BlockType.WOODER_DOOR.value)
    for door in doors:
        print(f"Open door at {door}")
        action = TeleportAndFaceCell(door)
        after_state, step_cost = agent.do(action, env)
        diff = after_state.diff(state)
        print_diff(diff)

def test_helper_functions():
    ''' Test some of the helper functions '''
    test_path = pathlib.Path(settings.ROOT_PATH) / "tests"
    with open(test_path / "polycraft_obs_tp_agent.p", "rb") as in_file:
        obs = pickle.load(in_file)

    state = obs.states[1]
    item_to_recipe = get_recipe_tree(state, poly.ItemType.WOODEN_POGO_STICK)
    assert(poly.ItemType.WOODEN_POGO_STICK.value in item_to_recipe)
    print(" ")
    print_recipe_tree(item_to_recipe, poly.ItemType.WOODEN_POGO_STICK)

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