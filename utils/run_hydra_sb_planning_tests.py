import settings
import worlds.science_birds as sb
from agent.hydra_agent import HydraAgent
import sys

def process_log(log):
    '''
    1. Count the number of default actions taken
    2. Count the amount of time spent planning for non default actions
    3. Score?
    '''
    current_level = 1
    ret = []
    defaults = 0
    with open(log,'r') as fin:
        while True:
            line = fin.readline()
            if not line:
                break
            if 'Problem simplification' in line:
                ret.append([current_level,float(line.split()[-1])])
            if '__get_default_action' in line:
                defaults += 1
            if 'WIN' in line or 'LOSS' in line:
                current_level += 1
    print(defaults)
    for value in ret:
        print('{},{}'.format(value[0],value[1]))


def main(config='30_level_0_novelties.xml',simplification_code=[0]):
    settings.DEBUG=False
    settings.SB_DEV_MODE=False
    settings.SB_PLANNER_SIMPLIFICATION_SEQUENCE = simplification_code
    settings.HEADLESS = True
    settings.SB_TIMEOUT = 180 # 3 minutes
    env = sb.ScienceBirds(None,launch=True,config=config)
    hydra = HydraAgent(env)
    hydra.main_loop()
    env.kill()

if __name__ == '__main__':
    main(simplification_code=[0,1])
    process_log('/home/klenk/PycharmProjects/hydra/tmp/tmp.log')