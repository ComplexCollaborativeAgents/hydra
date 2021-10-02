
import worlds.polycraft_world as poly
import pytest
from os import path
import settings

from agent.polycraft_hydra_agent import PolycraftHydraAgent
import logging

logging.basicConfig(format='%(name)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TestPolycraft")

@pytest.fixture(scope="module")
def launch_polycraft():
    logger.info("starting")

    env = poly.Polycraft(launch=True)
    yield env
    logger.info("teardown tests")
    env.kill()


# @pytest.mark.skip()
def test_polycraft(launch_polycraft: poly.Polycraft):
    env = launch_polycraft
    # env.sb_client.set_game_simulation_speed(settings.SB_SIM_SPEED)
    hydra = PolycraftHydraAgent()

    test_level = path.join(settings.ROOT_PATH, "bin", "pal", "pogo_100_PN", "POGO_L00_T01_S01_X0100_U9999_V0_G00000_I0020_N0.json")

    env.init_selected_level(test_level)

    state = env.get_current_state()

    logger.info("Initial state: {}".format(str(state)))

    action = hydra.choose_action(state)

    # Perform a set of actions
    for _ in range(50):
        logger.info("Chose action: {}".format(action))

        after_state, step_cost = env.act(action)

        logger.info("Post action state: {}".format(str(after_state)))
        logger.info("Post action step cost: {}".format(step_cost))
