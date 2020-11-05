import worlds.science_birds as sb
from agent.repairing_hydra_agent import RepairingHydraSBAgent
import argparse
from utils.host import Host


def parse():
    parser = argparse.ArgumentParser(description='HYDRA agent.')
    parser.add_argument('--server', type=Host.type, help='server host (format: hostname:port)')
    # parser.add_argument('--observer', type=Host.type, help='observer host (format: hostname:port)')
    arguments = parser.parse_args()
    return arguments


def main():
    arguments = parse()
    env = sb.ScienceBirds(None, launch=False, host=arguments.server)
    hydra = RepairingHydraSBAgent(env)
    hydra.main_loop()


if __name__ == '__main__':
    main()
