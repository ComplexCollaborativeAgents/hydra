
from worlds.polycraft_world import *
import pytest
from os import path
import settings
from utils.polycraft_utils import *
from agent.polycraft_hydra_agent import *
import logging

logging.basicConfig(format='%(name)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TestPolycraft")

@pytest.fixture(scope="module")
def launch_polycraft():
    logger.info("starting")
    env = Polycraft(polycraft_mode=ServerMode.SERVER)
    yield env
    logger.info("teardown tests")
    env.kill()

def test_polycraft_hydra(launch_polycraft: Polycraft):
    ''' Connect to polycraft and perform actions '''

    env = launch_polycraft
    hydra = PolycraftHydraAgent()

    test_level = path.join(settings.POLYCRAFT_NON_NOVELTY_LEVEL_DIR, "POGO_L00_T01_S01_X0100_U9999_V0_G00000_I0020_N0.json")

    env.init_selected_level(test_level)

    state = env.get_current_state()

    logger.info("Initial state: {}".format(str(state)))


    # Perform a set of actions
    for _ in range(50):
        action = hydra.choose_action(state)

        logger.info("Chose action: {}".format(action))

        after_state, step_cost = env.act(state, action)

        logger.info("Post action state: {}".format(str(after_state)))
        logger.info("Post action step cost: {}".format(step_cost))
        state = after_state
