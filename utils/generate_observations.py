import settings
import worlds.science_birds as sb
from agent.sb_hydra_agent import SBHydraAgent
import os.path as path
import os
import shutil
import argparse
from zipfile import ZipFile

USE_RANDOM_SHOT = False
USE_PLANNED_SHOT = True
USE_DEFAULT_SHOT = True

def main(config:str):
    """ Do a single round of generation for a particular config file """
    env = sb.ScienceBirds(launch=True,config=config)
    hydra = SBHydraAgent(env)
    hydra.main_loop()
    env.kill()


if __name__ == '__main__':
    path_prefix = "Phase2"

    # By default, first one in the path_prefix directory
    config_files = [os.listdir(path.join(settings.ROOT_PATH, "data", "science_birds", "config", path_prefix))[0]]

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', type=str, help='path prefix after hydra/data/science_birds/config', default=None, dest='path_prefix')
    parser.add_argument('-c', type=str, nargs='+', help='list of config files to run', default=None, dest='config_files')
    parser.add_argument('--use_random', action='store_true', help='use random shots when generating observations', dest='use_random')
    parser.add_argument('--no_random', action='store_false', help='do not use random shots when generating observations', dest='use_random')
    parser.add_argument('--use_default', action='store_true', help='use default shots when generating observations', dest='use_default')
    parser.add_argument('--no_default', action='store_false', help='do not use default shots when generating observations', dest='use_default')
    parser.add_argument('--use_planned', action='store_true', help='use planned shots when generating observations', dest='use_planned')
    parser.add_argument('--no_planned', action='store_false', help='do not use planned shots when generating observations', dest='use_planned')
    parser.set_defaults(use_random=False, use_planned=True, use_default=True)

    args = parser.parse_args()

    if args.path_prefix is not None:
        print("Using args path_prefix: {}".format(args.path_prefix))
        path_prefix = args.path_prefix
    if args.config_files is not None:
        print("Using args config_files: {}".format(args.config_files))
        config_files = args.config_files
    if args.use_random is not None:
        print("Using args random shot ({})...".format(args.use_random))
        USE_RANDOM_SHOT = args.use_random
    if args.use_default is not None:
        print("Using args default shot ({})...".format(args.use_default))
        USE_DEFAULT_SHOT = args.use_default
    if args.use_planned is not None:
        print("Using args planned shot ({})...".format(args.use_planned))
        USE_PLANNED_SHOT = args.use_planned

    trace_dir = path.join(settings.ROOT_PATH,"agent","consistency","trace","observations")

    # Iterate through each provided config file
    for config_file in config_files:
        print("Generating observations for {}".format(config_file))
        output_dir = path.join(settings.ROOT_PATH, "data", "science_birds", config_file[:-4])

        if not path.isdir(output_dir):
            os.mkdir(output_dir)

        if USE_DEFAULT_SHOT:
            print("Generating default shot observations...")
            settings.SB_DEFAULT_SHOT = ''
            main(config=path.join(path_prefix, config_file))

            copy_path_default =  path.join(output_dir, "default")

            if os.path.isdir(copy_path_default):
                # Move each file from staging directory to the existing directory
                for filename in os.listdir(trace_dir):
                    shutil.move(os.path.join(trace_dir, filename), os.path.join(copy_path_default, filename))
            else:
                # Create new directory
                shutil.copytree(trace_dir, copy_path_default)

        if USE_RANDOM_SHOT:
            print("Generating random shot observations...")
            settings.SB_DEFAULT_SHOT = 'RANDOM'
            main(config=path.join(path_prefix, config_file))

            copy_path_random = path.join(output_dir, "random")

            if os.path.isdir(copy_path_random):
                # Move each file from staging directory to the existing directory
                for filename in os.listdir(trace_dir):
                    shutil.move(os.path.join(trace_dir, filename), os.path.join(copy_path_random, filename))
            else:
                # Create new directory
                shutil.copytree(trace_dir, copy_path_random)

        if USE_PLANNED_SHOT:
            print("Generating planned shot observations...")
            main(config=path.join(path_prefix, config_file))

            copy_path_planned = path.join(output_dir, "planned")

            if os.path.isdir(copy_path_planned):
                # Move each file from staging directory to the existing directory
                for filename in os.listdir(trace_dir):
                    shutil.move(os.path.join(trace_dir, filename), os.path.join(copy_path_planned, filename))
            else:
                # Create new directory
                shutil.copytree(trace_dir, copy_path_planned)
            shutil.rmtree(trace_dir)

    print("Done.")