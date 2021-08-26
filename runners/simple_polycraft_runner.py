import argparse
from worlds.polycraft_world import *

import settings
import logging
import pathlib

LOG_PATH = pathlib.Path(settings.ROOT_PATH) / 'runners' / 'log' / 'hydra-sb.log'


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

    run_config = {
        "headless": False,
        "trial_path": os.path.join("..", "..", "polycraft_trials", "POGO_10game_prenovelty", "POGO_L00_T01_S01", "X0010", "POGO_L00_T01_S01_X0010_A_U9999_V0"),
    }

    env = Polycraft(launch=True, server_config=run_config)
    # hydra = RepairingHydraSBAgent(env)
    # hydra.main_loop()

    env.kill()


if __name__ == '__main__':
    main()
