
# from tests.test_polycraft import launch_polycraft
import pytest

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

@pytest.mark.parametrize('execution_number', range(3))
def test_debug(launch_polycraft, execution_number):
    ''' A bug in which an accessible trader is not accessible'''
    bug_level = "POGO_L00_T01_S01_X0100_U9999_V0_G00066_I0066_N0"
    levels = get_non_novelty_levels_files()
    test_level = [level for level in levels if bug_level in level.name][0]

    env, agent, levels = launch_polycraft
    logger.info(f"Loading level {test_level}...")
    env.init_selected_level(test_level)
    agent.start_level(env)  # Collect trades and recipes
    assert(True)

@pytest.mark.parametrize('execution_number', range(100))
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

@pytest.mark.parametrize('execution_number', range(1))
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
    state, step_cost = agent.do_batch(max_iterations, state, env)

    assert(state.count_items_of_type(ItemType.WOODEN_POGO_STICK.value)>0)
    logger.info("Pogo stick created !!!!")


@pytest.mark.parametrize('execution_number', range(1))
def test_expore_door(launch_polycraft, execution_number):
    ''' Test finding a plan to open the door and opening it '''
    env, agent, levels = launch_polycraft
    test_level = levels[execution_number % len(levels)]  # Choose the level to try now

    logger.info(f"Loading level {test_level}...")
    env.init_selected_level(test_level)
    agent.start_level(env)  # Collect trades and recipes
    state = env.get_current_state()

    assert(state.terminal==False)
    max_iterations = 30
    door_cells = state.get_cells_of_type(BlockType.WOODER_DOOR.value)
    if len(door_cells)==0:
        logger.info(f"No doors in level {test_level.name}, skipping test")
        return
    door_cell = door_cells[0]
    open_door_task = PolycraftTask.OPEN_DOOR.create_instance()
    open_door_task.door_cell = door_cell
    meta_model = PolycraftMetaModel(active_task=open_door_task)
    agent.set_meta_model(meta_model)
    max_iterations = 5
    old_num_of_known_cells = len(state.game_map)
    for i in range(max_iterations):
        action = agent.choose_action(state)
        state, _ = agent.do(action, env)
        new_num_of_known_cells = len(state.game_map)
        if new_num_of_known_cells > old_num_of_known_cells:
            return
    assert(False)

