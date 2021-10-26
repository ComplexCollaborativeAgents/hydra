import settings
import pathlib
import time
import sys

# Add HYDRA repo directory to PYTHONPATH
sys.path.insert(0, settings.ROOT_PATH)

from worlds.polycraft_world import Polycraft
from agent.polycraft_hydra_agent import PolycraftHydraAgent, FixedPlanPlanner

"""
Runner intended to interface with UTD's LaunchTournament.py (can be found in pal/PolycraftAIGym)
LaunchTournament.py handles trial sets and most of simulation management, such as loading next levels
"""

USE_STANDALONE = False   # For testing purposes, load levels (Not using LaunchTournament.py.  Only start when polycraft server is ready!)
# STANDALONE_LEVEL = pathlib.Path(settings.POLYCRAFT_DIR) / "available_tests" / "easy_pogo_lesson_1.json"
STANDALONE_LEVEL = pathlib.Path(settings.POLYCRAFT_NON_NOVELTY_LEVEL_DIR) / "POGO_L00_T01_S01_X0100_U9999_V0_G00000_I0020_N0.json"

def run():
    world = Polycraft(launch=False)

    agent = PolycraftHydraAgent(planner=FixedPlanPlanner())

    if USE_STANDALONE:
        world.init_selected_level(STANDALONE_LEVEL)
        time.sleep(12)

    is_running = True
    advancing_level = True

    # Start by sending a command over to signal agent ready (and get recipes)
    world.poly_client.CHECK_COST()

     # act
    state = world.get_current_state()

    while is_running:

        # Handle level change
        if advancing_level:
            # assert(len(world.current_recipes) == 0 and len(world.current_trades) == 0)
            # world.poly_client._logger.info("Trades before are: {}".format(world.current_trades))
            agent.start_level(world)
            state = world.get_current_state()
            # world.poly_client._logger.info("Trades after are: {}".format(world.current_trades))
            advancing_level = False      

        action = agent.choose_action(state)
        state, reward = agent.do(action, world)

        world.poly_client._logger.info("State: {}\nReward: {}".format(state, reward))

        # LaunchTournament.py handles detecting and advancing to next level
        if state.is_terminal():
            # Clean up old recipes and trades
            world.poly_client._logger.info("Finished prior level, preparing for new one")
            world.reset_current_trades()
            world.reset_current_recipes()
            advancing_level = True

if __name__ == "__main__":
    run()