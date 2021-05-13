import settings
import worlds.science_birds as sb
from agent.hydra_agent import HydraAgent
import os.path as path
import shutil


def main(config='all_level_0_novelties.xml'):
    settings.DEBUG=True
    settings.SB_DEV_MODE=False
    settings.NO_PLANNING=True
    settings.NO_REPAIR=True

    env = sb.ScienceBirds(None,launch=True,config=config)
    hydra = HydraAgent(env)
    hydra.main_loop()
    env.kill()

def eval_m18_data():
    """ Generates observation data for M18 evaluation """

    # References config files from data/science_birds/m18
    path_prefix = "M18"
    config_files = ["100_level_3_type_7_novelties.xml"] 

    for config_file in config_files:
        settings.SB_DEFAULT_SHOT = ''
        main(config=path.join(path_prefix, config_file))
        settings.SB_DEFAULT_SHOT = 'RANDOM'
        main(config=path.join(path_prefix, config_file))

        # Copies the observations (which are dumped into agent/consistency/trace/observations) into its own file
        # The observations from each config file are copied into their own directories under data/science_birds
        shutil.copytree(path.join(settings.ROOT_PATH,"agent","consistency","trace"),
                        path.join(settings.ROOT_PATH, "data", "science_birds", config_file[:-4]))

        shutil.rmtree(path.join(settings.ROOT_PATH,"agent","consistency","trace"))

if __name__ == '__main__':
    path_prefix = "M18"
    config_files = ["100_level_3_type_7_novelties.xml"]

    for config_file in config_files:
        settings.SB_DEFAULT_SHOT = ''
        main(config=path.join(path_prefix, config_file))
        settings.SB_DEFAULT_SHOT = 'RANDOM'
        main(config=path.join(path_prefix, config_file))

        shutil.copytree(path.join(settings.ROOT_PATH,"agent","consistency","trace"),
                        path.join(settings.ROOT_PATH, "data", "science_birds", config_file[:-4]))

        shutil.rmtree(path.join(settings.ROOT_PATH,"agent","consistency","trace"))