
import pickle
import pytest
import settings
import logging
import pathlib

from utils.polycraft_utils import *
from agent.planning.polycraft_planning.actions import *
from worlds.polycraft_world import *
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
    levels = []
    levels_dir_path = pathlib.Path(settings.POLYCRAFT_NON_NOVELTY_LEVEL_DIR)
    for level_file in os.listdir(levels_dir_path):
        if level_file.endswith(".json2")==False:
            levels.append(levels_dir_path / level_file)

    logger.info("starting")
    env = Polycraft(launch=True)
    agent = PolycraftHydraAgent()

    yield env, agent, levels

    logger.info("teardown tests")
    env.kill()

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

@pytest.mark.parametrize('execution_number', range(100))
def test_pddl_planner(launch_polycraft, execution_number):
    ''' Run the fixed planner and observe results '''
    env, agent, levels = launch_polycraft
    test_level = random.choice(levels)  # Choose the level to try now

    logger.info(f"Loading level {test_level}...")
    env.init_selected_level(test_level)
    agent.start_level(env)  # Collect trades and recipes
    state = env.get_current_state()

    assert(state.terminal==False)
    max_iterations = 30
    state, step_cost = agent.do_batch(max_iterations, state, env)

    assert(state.count_items_of_type(ItemType.WOODEN_POGO_STICK.value)>0)
    logger.info("Pogo stick created !!!!")