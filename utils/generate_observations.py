import settings
import worlds.science_birds as sb
from agent.sb_hydra_agent import SBHydraAgent
import os.path as path
import shutil
<<<<<<< Updated upstream
=======
import argparse
from zipfile import ZipFile
>>>>>>> Stashed changes


def main(config='all_level_0_novelties.xml'):
    settings.DEBUG=True
    settings.SB_DEV_MODE=False
    settings.NO_PLANNING=True
    settings.NO_REPAIR=True

    env = sb.ScienceBirds(None,launch=True,config=config)
    hydra = SBHydraAgent(env)
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

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', type=str, help='path prefix after hydra/data/science_birds/config', default=None, dest='path_prefix')
    parser.add_argument('-c', type=str, nargs='+', help='list of config files to run', default=None, dest='config_files')

    args = parser.parse_args()

    if args.path_prefix is not None:
        print("Using args path_prefix: {}".format(args.path_prefix))
        path_prefix = args.path_prefix
    if args.config_files is not None:
        print("Using args config_files: {}".format(args.config_files))
        config_files = args.config_files

    for config_file in config_files:
        settings.SB_DEFAULT_SHOT = ''
        main(config=path.join(path_prefix, config_file))

        copy_path_baseline = path.join(settings.ROOT_PATH, "data", "science_birds", config_file[:-4], "baseline")

        shutil.copytree(path.join(settings.ROOT_PATH,"agent","consistency","trace"),
                        copy_path_baseline)
        shutil.rmtree(path.join(settings.ROOT_PATH,"agent","consistency","trace"))

        settings.SB_DEFAULT_SHOT = 'RANDOM'
        main(config=path.join(path_prefix, config_file))

        copy_path_random = path.join(settings.ROOT_PATH, "data", "science_birds", config_file[:-4], "random")

        shutil.copytree(path.join(settings.ROOT_PATH,"agent","consistency","trace"),
                        copy_path_random)

        shutil.rmtree(path.join(settings.ROOT_PATH,"agent","consistency","trace"))