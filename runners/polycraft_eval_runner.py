import settings
import pathlib
import time
import sys

# Add HYDRA repo directory to PYTHONPATH
sys.path.insert(0, settings.ROOT_PATH)

from worlds.polycraft_world import Polycraft, ServerMode
from agent.polycraft_hydra_agent import PolycraftHydraAgent
"""
Runner intended to interface with UTD's LaunchTournament.py (can be found in pal/PolycraftAIGym)
LaunchTournament.py handles trial sets and most of simulation management, such as loading next levels
"""

RUNNER_MODE = ServerMode.CLIENT

SINGLE_LEVEL_MODE = False   # For testing purposes, load a single level and finish when it's done
SINGLE_LEVEL_TO_RUN = pathlib.Path(settings.POLYCRAFT_NON_NOVELTY_LEVEL_DIR) / "POGO_L00_T01_S01_X0100_U9999_V0_G00066_I0366_N0.json"

def setup_for_new_level(agent, world):
    world.init_state_information()
    agent.start_level(world)
    state = world.get_current_state()
    return state

def run():
    world = Polycraft(polycraft_mode=RUNNER_MODE)
    agent = PolycraftHydraAgent()

    if SINGLE_LEVEL_MODE:
        world.init_selected_level(SINGLE_LEVEL_TO_RUN)
        time.sleep(12)

    is_running = True

    # Start by sending a command over to signal agent ready (and get recipes)
    world.poly_client.CHECK_COST()

    # act
    state = setup_for_new_level(agent, world)
    current_step_num = state.step_num

    while is_running:

        # Handle level change
        if state.step_num < current_step_num:
            world.poly_client._logger.info(
                f"State num mismatch ({state.step_num}<{current_step_num}) -> starting a new level...")
            state = setup_for_new_level(agent, world)
            current_step_num = state.step_num

        action = agent.choose_action(state)
        state, reward = agent.do(action, world)

        agent.novelty_detection(report_novelty=True, only_current_state=True)

        world.poly_client._logger.info("State: {}\nReward: {}".format(state, reward))

        # LaunchTournament.py handles detecting and advancing to next level
        if state.is_terminal():
            # agent.novelty_detection(report_novelty=True) #
            if SINGLE_LEVEL_MODE:
                world.poly_client._logger.info("Finished the level!")
                return
            else:
                # Clean up old recipes and trades
                world.poly_client._logger.info("Finished prior level, preparing for new one")
                state = setup_for_new_level(agent, world)

if __name__ == "__main__":
    run()