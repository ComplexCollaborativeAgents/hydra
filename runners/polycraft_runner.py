from runners.polycraft_dispatcher import PolycraftDispatcher
from agent.polycraft_hydra_agent import PolycraftHydraAgent
from agent.planning.polycraft_planning.fixed_planner import FixedPlanPlanner
import argparse
from utils.host import Host
from worlds.polycraft_world import *

import settings
import logging
import pathlib

LOG_PATH = pathlib.Path(settings.ROOT_PATH) / 'runners' / 'log' / 'hydra-polycraft.log'


def config_logging():
    logger = logging.getLogger()
    fh = logging.FileHandler(LOG_PATH)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def parse():
    parser = argparse.ArgumentParser(description='HYDRA agent.')
    parser.add_argument('--server', type=Host.type, help='server host (format: hostname:port)')
    # parser.add_argument('--observer', type=Host.type, help='observer host (format: hostname:port)')
    arguments = parser.parse_args()
    return arguments


def main():
    config_logging()
    arguments = parse()

    # trials = [os.path.join(settings.POLYCRAFT_LEVEL_DIR, "POGO_10game_prenovelty", "POGO_L00_T01_S01", "X0010", "POGO_L00_T01_S01_X0010_A_U9999_V0")]
    trials = [os.path.join(settings.POLYCRAFT_NON_NOVELTY_LEVEL_DIR)]

    agent = PolycraftHydraAgent(planner=FixedPlanPlanner())

    dispatcher = PolycraftDispatcher(agent=agent)
    try:
        dispatcher.experiment_start(trials=trials, standalone=False)

        dispatcher.run_trials()
    finally:
        dispatcher.experiment_end() # Important to clean up memory and connection to polycraft server

if __name__ == '__main__':
    main()
