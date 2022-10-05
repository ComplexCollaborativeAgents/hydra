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

RUNNER_MODE = ServerMode.TOURNAMENT

SINGLE_LEVEL_MODE = False  # For testing purposes, load a single level and finish when it's done
SINGLE_LEVEL_TO_RUN = pathlib.Path(
    settings.POLYCRAFT_NON_NOVELTY_LEVEL_DIR) / "POGO_L00_T01_S01_X0100_U9999_V0_G00066_I0366_N0.json"

class PolycraftTournamentRunner:
    def __init__(self, runner_mode:ServerMode, single_level_mode:bool):
        """"""
        self.world = Polycraft(polycraft_mode=runner_mode)
        self.agent = PolycraftHydraAgent()

        self.single_level_mode=single_level_mode

    def setup_for_new_level(self):
        self.world.init_state_information()
        self.agent.episode_init(self.world)
        return self.world.get_current_state()

    def run(self):
        if self.single_level_mode:
            self.world.init_selected_level(SINGLE_LEVEL_TO_RUN)
            time.sleep(12)

        is_running = True

        # Start by sending a command over to signal agent ready (and get recipes)
        self.world.poly_client.CHECK_COST()

        # act
        state = self.setup_for_new_level()
        current_step_num = state.step_num

        while is_running:

            try:

                # Handle level change
                if state.step_num < current_step_num:
                    self.world.poly_client._logger.info(
                        f"State num mismatch ({state.step_num}<{current_step_num}) -> starting a new level...")
                    
                    self.world.poly_client._logger.info("Waiting for level to load...")
                    time.sleep(10.0) # Wait for new level to load
                    self.world.poly_client._logger.info("Performing exploratory actions...")
                    state = self.setup_for_new_level()
                    current_step_num = state.step_num

                action = self.agent.choose_action(state, self.world)
                state, reward = self.agent.do(action, self.world)

                self.world.poly_client._logger.info("State: {}\nReward: {}".format(state, reward))

                # LaunchTournament.py handles detecting and advancing to next level
                if state.is_terminal():
                    self.agent.novelty_detection(report_novelty=True)
                    if SINGLE_LEVEL_MODE:
                        self.world.poly_client._logger.info("Finished the level!")
                        return
                    else:
                        # Clean up old recipes and trades
                        self.world.poly_client._logger.info("Finished prior level, preparing for new one")
                        state = self.setup_for_new_level()
            except Exception as e:
                self.world.poly_client._logger.info(f"Something made the agent crash! {str(e)}\n{str(e.__traceback__)}")
                self.world.poly_client.REPORT_NOVELTY(level="1", confidence="100",
                                                user_msg='Agent crashed, probably an unknown object. ')
                self.world.poly_client.GIVE_UP()
                self.world.poly_client.CHECK_COST()
                state = self.setup_for_new_level()
                raise e


if __name__ == "__main__":
    runner = PolycraftTournamentRunner(RUNNER_MODE, SINGLE_LEVEL_MODE)
    runner.run()
