import worlds.science_birds as sb
from agent.repairing_hydra_agent import RepairingHydraSBAgent
import argparse
from utils.host import Host

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
    env = sb.ScienceBirds(None, launch=False, host=arguments.server)
    hydra = RepairingHydraSBAgent(env)
    hydra.main_loop()


if __name__ == '__main__':
    main()
