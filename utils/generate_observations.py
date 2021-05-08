import settings
import worlds.science_birds as sb
from agent.hydra_agent import HydraAgent
import sys


def main(config='all_level_0_novelties.xml'):
    settings.DEBUG=True
    settings.SB_DEV_MODE=False
    settings.NO_PLANNING=True
    env = sb.ScienceBirds(None,launch=True,config=config)
    hydra = HydraAgent(env)
    hydra.main_loop()
    env.kill()

if __name__ == '__main__':
    print(len(sys.argv))
    print(sys.argv)
    if len(sys.argv)>=2 and sys.argv[1]=='--random':
        settings.SB_DEFAULT_SHOT = 'RANDOM'
    if len(sys.argv)>=4 and sys.argv[2]=='--config-file':
        config_file = sys.argv[3].strip()
    main(config =config_file)