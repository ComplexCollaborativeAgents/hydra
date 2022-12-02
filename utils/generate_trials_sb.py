import pathlib
import logging
import random
import os
import xml.etree.ElementTree as ET

from typing import Dict, List, Tuple

import settings

logging.basicConfig(format='%(name)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("generate_trials")
logger.setLevel(logging.INFO)

# Types and levels of novelty
# NOVELTY_LEVELS = [
#     'novelty_level_1', 'novelty_level_2', 'novelty_level_3',
#     'novelty_level_11', 'novelty_level_12', 'novelty_level_13',
#     'novelty_level_22', 'novelty_level_23', 'novelty_level_24', 'novelty_level_25'
# ]
NOVELTY_LEVELS = [
    'novelty_level_11', 'novelty_level_12', 'novelty_level_13',
    'novelty_level_24', 'novelty_level_25',
    'novelty_level_36', 'novelty_level_37', 'novelty_level_38',
]
NON_NOVEL_LEVELS = ['novelty_level_0']

# Constants
SB_CONFIG_PATH = pathlib.Path(settings.ROOT_PATH) / 'data' / 'science_birds' / 'config'
CONFIG_TEMPLATE = SB_CONFIG_PATH / 'test_config.xml'    # Template to build subsequent xml config files off of
SB_BIN_PATH = pathlib.Path(settings.SCIENCE_BIRDS_BIN_DIR) / 'linux'
SB_LEVELS_PATH = pathlib.Path(settings.SCIENCE_BIRDS_LEVELS_DIR)   # Path to the Science Birds application directory
NON_NOVELTY_DIR = os.path.join('Levels', 'novelty_level_0')

# Settings
NUM_TRIALS = 2      # Number of trials to generate
NUM_LEVELS = 40     # Levels per trial
LEVELS_BEFORE_NOVELTY = 40   # Levels before novelty is introduced
NOTIFY_NOVELTY = True
REPETITION = 10   # If not set to None, the same sampled level will be used this many times before another is selected.
NON_NOVEL_TO_USE = { # level and type of non-novel levels to use
    'novelty_level_0': [
        "type222", "type223", "type224", "type225", "type226", "type227",
        "type232", "type233", "type234", "type235", "type236", "type237",
        "type242", "type243", "type244", "type245", "type246", "type247",
        "type252", "type253", "type254", "type255", "type256", "type257"
        ]
    }   
NOVEL_TO_USE = {    # level and type of novel levels to use
    'novelty_level_1': [
        'type6'
    ]
}   
OUTPUT_DIR = SB_CONFIG_PATH / 'Phase2'

def format_trial_id(trial_num: int, num_levels: int, novelty: str, n_type: str, nn_type: str) -> str:
    """
    Format trial parameters into a semi-unique identifier
    """
    return '{}_{}_{}_{}_non-novelty_{}'.format(trial_num, num_levels, novelty, n_type, nn_type)


def unpack_trial_id(trial_id: str) -> Tuple:
    """
    Unpack a trial_id to get the parameters of the trial
    :returns: Tuple of size 5:
        1) trial_num: The trial number
        2) num_levels: Number of levels per trial
        3) novelty: Novelty level used in the trial
        4) n_type: Novelty type used in the trial
        5) nn_type: Non-novelty type used in the trial
    """
    segments = trial_id.split('_')

    trial_num = int(segments[0])
    num_levels = int(segments[1])
    novelty = segments[2]
    n_type = segments[3]
    nn_type = segments[5]

    return trial_num, num_levels, novelty, n_type, nn_type

def find_novelty_levels(exclude_non_novel:bool=False) -> List[str]:
    """Finds a list of novelty levels that exist

    Args:
        exclude_non_novel (bool): excludes non-novel levels

    Returns:
        List[str]: list of novelty levels
    """
    levels = os.listdir(settings.SCIENCE_BIRDS_LEVELS_DIR)

    if exclude_non_novel:
        levels = [l for l in levels if l not in NON_NOVEL_LEVELS]

    return levels

def find_novelty_types(level:str=None) -> List[str]:
    """Finds a list of novelty types from each novelty level

    Args:
        level (str): Optional.  If level is specified, will only retrieve novelty types for a specified novelty level

    Returns:
        List[str]: List of novelty types
    """

    types_list = []

    for novelty_lvl in find_novelty_levels():
        if level is not None and level != novelty_lvl:
            continue
        dirpath = os.path.join(settings.SCIENCE_BIRDS_LEVELS_DIR, novelty_lvl)
        types_list.extend(os.listdir(dirpath))

    return types_list

def collect_levels() -> Dict[str, List[str]]:
    """
    Collects levels from the directory of available levels
    """
    all_levels = {}

    # Iterate over all non novel levels
    for non_novelty_level in NON_NOVEL_LEVELS:
        all_levels[non_novelty_level] = {}

        logger.info("Loading levels from non-novelty level {}".format(non_novelty_level))
        # NOTE: SB makes use of relative paths to each level from the root SB directory
        non_novel_dirs = os.listdir(os.path.join(SB_LEVELS_PATH, non_novelty_level))

        # Collect non novelty levels
        for non_novel_type in find_novelty_types(non_novelty_level):
            if non_novel_type not in non_novel_dirs:
                # Skip over types not present
                logger.debug("non-novel level {} has no type: {}".format(non_novelty_level, non_novel_type))
                continue

            pattern = './9001_Data/StreamingAssets/Levels/{}/{}/Levels/*.xml'.format(non_novelty_level, non_novel_type)
            levels = list(SB_BIN_PATH.glob(pattern))
            all_levels[non_novelty_level][non_novel_type] = []

            if len(levels) == 0:
                raise ValueError("Unable to find non-novelty levels in {} using pattern {}".format(SB_LEVELS_PATH, pattern))

            for level in levels: # Iterate over all levels of the novelty level and typeå
                # Add level to set - use shorthand for easier access later
                all_levels[non_novelty_level][non_novel_type].append(level)

            logger.debug("Non novelty {} has {} levels".format(non_novel_type, len(all_levels[non_novelty_level][non_novel_type])))

    for novelty_level in find_novelty_levels(exclude_non_novel=True):   # Iterate over all novelty levels
        all_levels[novelty_level] = {}

        logger.info("Loading levels from novelty level {}".format(novelty_level))
        # NOTE: SB makes use of relative paths to each level from the root SB directory
        novelty_dirs = os.listdir(os.path.join(SB_LEVELS_PATH, novelty_level))

        for novelty_type in find_novelty_types(novelty_level):  # Iterate over all novelty types
            if novelty_type not in novelty_dirs:
                # Skip over types not present
                logger.debug("novel level {} has no type: {}".format(novelty_level, novelty_type))
                continue

            pattern = './9001_Data/StreamingAssets/Levels/{}/{}/Levels/*.xml'.format(novelty_level, novelty_type)
            levels = list(SB_BIN_PATH.glob(pattern))
            all_levels[novelty_level][novelty_type] = []

            if len(levels) == 0:
                raise ValueError("Unable to find non-novelty levels in {} using pattern {}".format(SB_LEVELS_PATH, pattern))

            if len(levels) == 0:
                raise ValueError("Unable to find novelty levels in {} using pattern {}".format(SB_LEVELS_PATH, pattern))

            for level in levels: # Iterate over all levels of the novelty level and typeå
                # Add level to set - use shorthand for easier access later
                all_levels[novelty_level][novelty_type].append(level)
            
            logger.debug("Novelty {} has {} levels".format(novelty_type, len(all_levels[novelty_level][novelty_type])))

    return all_levels


def generate_trial_sets(num_trials:int, num_levels:int, 
                        levels_before_novelty:int, repetition:int, 
                        non_novel_to_use:dict, novel_to_use:dict,
                        level_sets:dict) -> Dict[str, Dict[str, List[str]]]:
    """
    Generate multiple trials using the provided non-novelty levels/types and novelty levels/types.
    :param num_trials: Number of trials to generate
    :param num_levels: Number of levels per trial
    :param levels_before_novelty: Number of levels in a trial before novelty is introduced
    :param repetition: Number of times a level will be repeated within a trial
    :param non_novel_to_use: Dictionary with key:value pairs: non_novel_level:list_of_non_novel_types.  A trial will be generated for each non-novel level/type.
    :param novel_to_use: Dictionary with key:value pairs - novel_level:list_of_novel_types.  A trial will be generated for each novelty level/type.
    :param level_sets: Nested dictionary with key:value pairs - novelty_level:novelty_type:list_of_levels.  Contains all collected non-novel levels/types as well as novelty levels/types
    :return: Dictionary of lists - trial_id:list_of_levels.
    """
    trials = {}

    # Create trials
    for trial_num in range(num_trials):
        for novelty, novel_types in novel_to_use.items():
            for nn_type in non_novel_to_use['novelty_level_0']:
                # Create a trial for every type
                for n_type in novel_types:
                    trial_id = format_trial_id(trial_num, num_levels, novelty, n_type, nn_type)
                    
                    logger.debug("Generating trial for {}".format(trial_id))
                    trials[trial_id] = generate_trial(nn_type, novelty, n_type, 
                                                       num_levels, levels_before_novelty, 
                                                       repetition, level_sets, )

    return trials
        
def generate_trial(nn_type: str, novelty: str, n_type: str, 
                    num_levels: int, levels_before_novelty: int, repetition: int,
                    level_sets: dict) -> List[str]:
    """
    Constructs a single trial using the provided non-novelty levels/types and novelty levels/types.
    :param nn_type: Non-novelty type that the trial uses
    :param novelty: Novelty level that the trial uses
    :param n_type: Novelty type that the trial uses
    :param num_levels: Number of levels per trial
    :param levels_before_novelty: Number of levels in a trial before novelty is introduced
    :param repetition: Number of times a level will be repeated within a trial
    :param level_sets: Nested dictionary with key:value pairs - novelty_level:novelty_type:list_of_levels.  Contains all collected non-novel levels/types as well as novelty levels/types
    """
    trial = []
    
    to_add = level_sets['novelty_level_0'][nn_type][0] # placeholder
    for level_num in range(num_levels):
        # Use sampled non-novel level from set of non-novel levels to use
        if level_num < levels_before_novelty:
            # Update sampled level to add 
            # (if repetition is enabled, only update when repetition count is reached)
            if repetition is None or (repetition is not None and level_num % repetition == 0):
                index = random.choice(list(range(len(level_sets['novelty_level_0'][nn_type]))))
                to_add = level_sets['novelty_level_0'][nn_type].pop(index) 
        else:   # Use level sampled from novelty level / type
            # Update sampled level to add 
            # (if repetition is enabled, only update when repetition count is reached)
            if repetition is None or (repetition is not None and level_num % repetition == 0):
                index = random.choice(list(range(len(level_sets[novelty][n_type]))))
                to_add = level_sets[novelty][n_type].pop(index) 
        # finally append the level
        trial.append(to_add)

    return trial


def save_trial_to_file(trial_levels:list, notify_novelty:bool, output_filepath:str):
    """
    Saves a trial into an .xml config file
    :param trial: 
    :param notify_novelty: Whether or not the trial notifies the agent that novelty is present
    :param output_dir:
    """
    
    # Create xml config file for trial
    tree = ET.parse(CONFIG_TEMPLATE)

    # Set notify novelty flag if applicable
    if notify_novelty:
        xpath = './trials/trial'
        trial = tree.getroot().find(xpath)
        trial.set('notify_novelty', str(notify_novelty))

    xpath = './trials/trial/game_level_set'
    level_set = tree.getroot().find(xpath)
    level_set.set('time_limit', '500000')
    level_set.set('total_interaction_limit', '1000000')

    for child in list(level_set):
        level_set.remove(child)

    for level in trial_levels:
        relpath = os.path.relpath(level, SB_BIN_PATH)
        ET.SubElement(level_set, 'game_levels', level_path=relpath)

    tree.write(output_filepath)


def generate_eval_trial_sets():
    levels = collect_levels()   # Collect levels

    # Generate the trial sets
    trials = generate_trial_sets(NUM_TRIALS, NUM_LEVELS, LEVELS_BEFORE_NOVELTY,
                                 REPETITION,
                                 NON_NOVEL_TO_USE, NOVEL_TO_USE,
                                 levels)

    # Save each trial into a file
    for trial_id, trial in trials.items(): 
        output_filepath = str(OUTPUT_DIR / "{}.xml".format(trial_id))
        save_trial_to_file(trial, NOTIFY_NOVELTY, output_filepath)

if __name__ == "__main__":
    generate_eval_trial_sets()
