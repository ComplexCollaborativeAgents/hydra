import settings
import worlds.science_birds as sb
from agent.hydra_agent import HydraAgent


def main(config='all_level_0_novelties.xml'):
    settings.DEBUG=True
    settings.SB_DEV_MODE=False
    settings.NO_PLANNING=True
    env = sb.ScienceBirds(None,launch=True,config=config)
    hydra = HydraAgent(env)
    hydra.main_loop()
    env.kill()

if __name__ == '__main__':
    main()