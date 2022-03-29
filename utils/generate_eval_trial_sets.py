import json
import pathlib
import random
import os
import xml.etree.ElementTree as ET

import settings

from collections import defaultdict

# Types and levels of novelty
NOVELTY_LEVELS = [
    'novelty_level_1', 'novelty_level_2', 'novelty_level_3',
    'novelty_level_11', 'novelty_level_12', 'novelty_level_13',
    'novelty_level_22', 'novelty_level_23', 'novelty_level_24', 'novelty_level_25'
]
NON_NOVEL_LEVELS = ['novelty_level_0']
NOVELTY_TYPES = [
    "type1", "type2", "type4", "type5", "type6", "type7", "type8", "type9", "type10", 
    "type12", "type14", "type15", "type20"
    "type22", "type23", "type24", "type25", "type50", "type90", "type110", "type130"
]
NON_NOVEL_TYPES = [
    "type2", "type4", "type5", "type6", "type7", 
    "type22", "type23", "type24", "type25", "type26",
    "type222", "type223", "type224", "type225", "type226", "type227",
    "type232", "type233", "type234", "type235", "type236", "type237",
    "type242", "type243", "type244", "type245", "type246", "type247",
    "type252", "type253", "type254", "type255", "type256", "type257"  
]

# Constants
SB_CONFIG_PATH = pathlib.Path(settings.ROOT_PATH) / 'data' / 'science_birds' / 'config'
CONFIG_TEMPLATE = SB_CONFIG_PATH / 'test_config.xml'    # Template to build subsequent xml config files off of
SB_BIN_PATH = pathlib.Path(settings.SCIENCE_BIRDS_BIN_DIR) / 'linux'    # Path to the Science Birds application directory
NON_NOVELTY_DIR = os.path.join('Levels', 'novelty_level_0')

# Settings
NUM_TRIALS = 2
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


if __name__ == "__main__":
    trials = {}
    all_levels = {}

    novelty_path = ""

    # Iterate over all non novel levels
    for non_novelty_level in NON_NOVEL_LEVELS:
        all_levels[non_novelty_level] = {}

        print("Loading levels from non-novelty level {}".format(non_novelty_level))
        non_novel_path = os.path.join("Levels", non_novelty_level)
        non_novel_dirs = os.listdir(os.path.join(SB_BIN_PATH, non_novel_path))

        # Collect non novelty levels
        for non_novel_type in NON_NOVEL_TYPES:
            if non_novel_type not in non_novel_dirs:
                # Skip over types not present
                print("non-novel level {} has no type: {}".format(non_novelty_level, non_novel_type))
                continue

            pattern = 'Levels/{}/{}/Levels/*.xml'.format(non_novelty_level, non_novel_type)
            levels = list(SB_BIN_PATH.glob(pattern))
            all_levels[non_novelty_level][non_novel_type] = []

            for level in levels: # Iterate over all levels of the novelty level and typeå
                # Add level to set - use shorthand for easier access later
                all_levels[non_novelty_level][non_novel_type].append(level)

            print("Non novelty {} has {} levels".format(non_novel_type, len(all_levels[non_novelty_level][non_novel_type])))

    for novelty_level in NOVELTY_LEVELS:   # Iterate over all novelty levels
        all_levels[novelty_level] = {}

        print("Loading levels from novelty level {}".format(novelty_level))
        
        novelty_path = os.path.join("Levels", novelty_level)
        novelty_dirs = os.listdir(os.path.join(SB_BIN_PATH, novelty_path))
        for novelty_type in NOVELTY_TYPES:  # Iterate over all novelty types
            if novelty_type not in novelty_dirs:
                # Skip over types not present
                print("novel level {} has no type: {}".format(novelty_level, novelty_type))
                continue

            # dir_path = SB_BIN_PATH / novelty_path / NOVELTY_TYPES[novelty_type] / "Levels"
            # print("SB: Loading levels from novelty type {} directory {}".format(novelty_type, dir_path))

            pattern = './Levels/{}/{}/Levels/*.xml'.format(novelty_level, novelty_type)
            levels = list(SB_BIN_PATH.glob(pattern))
            all_levels[novelty_level][novelty_type] = []

            for level in levels: # Iterate over all levels of the novelty level and typeå
                # Add level to set - use shorthand for easier access later
                all_levels[novelty_level][novelty_type].append(level)
            
            print("Novelty {} has {} levels".format(novelty_type, len(all_levels[novelty_level][novelty_type])))

    # Create trials
    for trial_num in range(NUM_TRIALS):
        for novelty, novel_types in NOVEL_TO_USE.items():
            for nn_type in NON_NOVEL_TO_USE['novelty_level_0']:
                # Create a trial for every type
                for n_type in novel_types:
                    trial_id = '{}_{}_{}_non-novelty_{}'.format(NUM_LEVELS, novelty, n_type, nn_type)
                    trials[trial_id] = []
                    output_filepath = OUTPUT_DIR / "{}.xml".format(trial_id)

                    print("Generating trial for {}".format(trial_id))

                    to_add = all_levels['novelty_level_0'][nn_type][0] # placeholder
                    for level_num in range(NUM_LEVELS):
                        # Use sampled non-novel level from set of non-novel levels to use
                        if level_num < LEVELS_BEFORE_NOVELTY:
                            # Update sampled level to add 
                            # (if repetition is enabled, only update when repetition count is reached)
                            if REPETITION is None or (REPETITION is not None and level_num % REPETITION == 0):
                                index = random.choice(list(range(len(all_levels['novelty_level_0'][nn_type]))))
                                to_add = all_levels['novelty_level_0'][nn_type].pop(index) 
                        else:   # Use level sampled from novelty level / type
                            # Update sampled level to add 
                            # (if repetition is enabled, only update when repetition count is reached)
                            if REPETITION is None or (REPETITION is not None and level_num % REPETITION == 0):
                                index = random.choice(list(range(len(all_levels[novelty][n_type]))))
                                to_add = all_levels[novelty][n_type].pop(index) 
                        # finally append the level
                        trials[trial_id].append(to_add)

                    # Create xml config file for trial
                    tree = ET.parse(CONFIG_TEMPLATE)

                    # Set notify novelty flag if applicable
                    if NOTIFY_NOVELTY:
                        xpath = './trials/trial'
                        trial = tree.getroot().find(xpath)
                        trial.set('notify_novelty', str(NOTIFY_NOVELTY))

                    xpath = './trials/trial/game_level_set'
                    level_set = tree.getroot().find(xpath)
                    level_set.set('time_limit', '500000')
                    level_set.set('total_interaction_limit', '1000000')

                    for child in list(level_set):
                        level_set.remove(child)

                    for level in trials[trial_id]:
                        relpath = os.path.relpath(level, SB_BIN_PATH)
                        ET.SubElement(level_set, 'game_levels', level_path=relpath)

                    tree.write(output_filepath)