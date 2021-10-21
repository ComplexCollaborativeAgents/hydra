
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

TEST_LEVEL = pathlib.Path(
    settings.POLYCRAFT_NON_NOVELTY_LEVEL_DIR) / "POGO_L00_T01_S01_X0100_U9999_V0_G00000_I0020_N0.json"

@pytest.fixture(scope="module")
def launch_polycraft():
    logger.info("starting")

    env = Polycraft(launch=True)
    agent = PolycraftHydraAgent()
    env.init_selected_level(TEST_LEVEL)
    agent.start_level(env)  # Collect trades and recipes
    state = env.get_current_state()

    yield env, agent, state

    logger.info("teardown tests")
    env.kill()

def test_fixed_planner(launch_polycraft):
    ''' Run the fixed planner and observe results '''
    env, agent, state = launch_polycraft

    assert(state.terminal==False)
    agent.planner = FixedPlanPlanner()
    while state.terminal==False and state.count_items_of_type(ItemType.WOODEN_POGO_STICK.value)==0:
        action = agent.choose_action(state)
        after_state, step_cost = agent.do(action, env)
        state = after_state

    assert(state.count_items_of_type(ItemType.WOODEN_POGO_STICK.value)>0)
    logger.info("Pogo stick created !!!!")