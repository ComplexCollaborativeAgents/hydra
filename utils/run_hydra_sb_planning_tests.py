import settings
import worlds.science_birds as sb
from agent.hydra_agent import HydraAgent
import sys

def main(config='all_level_0_novelties.xml',simplification_code=2):
    settings.DEBUG=False
    settings.SB_DEV_MODE=False
    settings.SB_PLANNER_SIMPLIFICATION_SEQUENCE = [simplification_code]
    settings.HEADLESS = True
    settings.SB_TIMEOUT = 180 # 3 minutes
    env = sb.ScienceBirds(None,launch=True,config=config)
    hydra = HydraAgent(env)
    hydra.main_loop()
    env.kill()

if __name__ == '__main__':
    if len(sys.argv)>1:
        main(sys.argv[1])
    else:
        main()