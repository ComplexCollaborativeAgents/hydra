
# from tests.test_polycraft import launch_polycraft
import pytest

import worlds.polycraft_world
from agent.planning.polycraft_planning.fixed_planner import FixedPlanPlanner
from agent.polycraft_hydra_agent import *

logging.basicConfig(format='%(name)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TestPolycraftWorld")
logger.setLevel(logging.INFO)

# TEST_LEVEL = pathlib.Path(
#     settings.POLYCRAFT_NON_NOVELTY_LEVEL_DIR) / "POGO_L00_T01_S01_X0100_U9999_V0_G00000_I0020_N0.json"

TESTS_LEVELS_DIR = pathlib.Path(settings.POLYCRAFT_NON_NOVELTY_LEVEL_DIR)

@pytest.fixture(scope="module")
def launch_polycraft():
    # Load levels
    levels = get_non_novelty_levels_files()
    logger.info("starting")
    env = Polycraft(polycraft_mode= ServerMode.SERVER)
    agent = PolycraftHydraAgent()

    yield env, agent, levels

    logger.info("teardown tests")
    env.kill()


@pytest.mark.parametrize('execution_number', range(1))
def test_fixed_planner(launch_polycraft, execution_number):
    ''' Run the fixed planner and observe results '''
    env, agent, levels = launch_polycraft
    test_level = random.choice(levels)  # Choose the level to try now

    logger.info(f"Loading level {test_level}...")
    env.init_selected_level(test_level)
    agent.start_level(env)  # Collect trades and recipes
    state = env.get_current_state()

    assert(state.terminal==False)
    agent.planner = FixedPlanPlanner()
    max_iterations = 30
    state, step_cost = agent.do_batch(max_iterations, state, env)

    assert(state.count_items_of_type(ItemType.WOODEN_POGO_STICK.value)>0)
    logger.info("Pogo stick created !!!!")

@pytest.mark.parametrize('execution_number', range(5))
def test_pddl_planner(launch_polycraft, execution_number):
    ''' Run the fixed planner and observe results '''
    env, agent, levels = launch_polycraft
    test_level = levels[execution_number % len(levels)]  # Choose the level to try now

    logger.info(f"Loading level {test_level}...")
    env.init_selected_level(test_level)
    agent.start_level(env)  # Collect trades and recipes
    state = env.get_current_state()

    assert(state.terminal==False)
    max_iterations = 30
    state, step_cost = agent.do_batch(max_iterations, state, env, time_limit=600)

    assert(state.count_items_of_type(ItemType.WOODEN_POGO_STICK.value)>0)
    logger.info("Pogo stick created !!!!")


@pytest.mark.parametrize('execution_number', range(5))
def test_pddl_planner_on_novelty_levels(launch_polycraft, execution_number):
    ''' Run the fixed planner and observe results '''
    env, agent, levels = launch_polycraft

    levels = get_novelty_levels_files()
    test_level = random.choice(levels)

    logger.info(f"Loading level {test_level}...")
    env.init_selected_level(test_level)
    agent.start_level(env)  # Collect trades and recipes
    state = env.get_current_state()

    assert(state.terminal==False)
    max_iterations = 30
    state, step_cost = agent.do_batch(max_iterations, state, env, time_limit=600)

    assert(state.count_items_of_type(ItemType.WOODEN_POGO_STICK.value)>0)
    logger.info("Pogo stick created !!!!")

@pytest.mark.parametrize('execution_number', range(1))
def test_explore_door(launch_polycraft, execution_number):
    ''' Test finding a plan to open the door and opening it '''
    env, agent, levels = launch_polycraft
    test_level = levels[execution_number % len(levels)]  # Choose the level to try now

    logger.info(f"Loading level {test_level}...")
    env.init_selected_level(test_level)
    agent.start_level(env)  # Collect trades and recipes
    state = env.get_current_state()
    door_cells = state.get_cells_of_type(BlockType.WOODER_DOOR.value)
    if len(door_cells)==0:
        logger.info(f"No doors in level {test_level.name}, skipping test")
        return
    for door_cell in door_cells:
        assert(_explore_door(door_cell, agent, env))
        logger.info(f"Door {door_cell} successfully explored!")

def _explore_door(door_cell:str, agent:PolycraftHydraAgent,env:Polycraft)->bool:
    ''' Run the agent to explore the given door. Return true iff successfully completed the task '''
    open_door_task = PolycraftTask.EXPLORE_DOOR.create_instance()
    open_door_task.door_cell = door_cell
    max_iterations = 5
    state = env.get_current_state()
    task_done = False
    agent.active_plan = agent.plan(active_task=open_door_task)
    for i in range(max_iterations):
        action = agent.choose_action(state)
        state, _ = agent.do(action, env)
        if len(state.door_to_room_cells[door_cell]) > 1:
            task_done = True
            assert(open_door_task.is_done(state))
            break
        else:
            assert (open_door_task.is_done(state)==False)
    return task_done

@pytest.mark.parametrize('execution_number', range(1))
def test_collect_from_safe(launch_polycraft, execution_number):
    ''' Test finding a plan to open the door and opening it '''
    env, agent, levels = launch_polycraft
    test_level = levels[execution_number % len(levels)]  # Choose the level to try now

    logger.info(f"Loading level {test_level}...")
    env.init_selected_level(test_level)
    agent.start_level(env)  # Collect trades and recipes
    state = env.get_current_state()
    door_cells = state.get_cells_of_type(BlockType.WOODER_DOOR.value)
    if len(door_cells)==0:
        logger.info(f"No doors in level {test_level.name}, skipping test")
        return

    safe_cells = []
    for door_cell in door_cells:
        task_done = _explore_door(door_cell, agent, env)
        assert(task_done)

        state = env.get_current_state()
        safe_cells = state.get_cells_of_type(BlockType.SAFE.value)
        if len(safe_cells)==0:
            logger.info(f"No safes in room {door_cell}, explore a different room")
            continue

    if len(safe_cells)==0:
        logger.info(f"No safes in level {test_level.name}, skipping test")
        return

    safe_cell = safe_cells[0]
    collect_safe_task = PolycraftTask.COLLECT_FROM_SAFE.create_instance()
    collect_safe_task.safe_cell = safe_cell
    agent.active_plan = agent.plan(active_task=collect_safe_task)
    max_iterations = 30
    diamonds_before = state.count_items_of_type(ItemType.DIAMOND.value)
    for i in range(max_iterations):
        action = agent.choose_action(state)
        state, _ = agent.do(action, env)
        diamonds_now = state.count_items_of_type(ItemType.DIAMOND.value)
        if diamonds_now-diamonds_before > 10: # TODO: REPLACE 10 WITH VALUE IN META MODEL
            logger.info(f"I think safe was collected! have {diamonds_now-diamonds_before} new diamonds!")
            return
    assert(False)
