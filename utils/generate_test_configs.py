import json
import pathlib
import random
import os
import xml.etree.ElementTree as ET

import settings

NOVELTIES = {'1': ['6', '7', '8', '9', '10'], '2': ['6', '7', '8', '9', '10'], '3': ['6', '7']} # Novelties to generate for
LEVELS_PER_NOVELTY = 3
SB_BIN_PATH = pathlib.Path(settings.SCIENCE_BIRDS_BIN_DIR) / 'linux'
SB_LEVEL_PATH = pathlib.Path(SB_BIN_PATH / 'Levels')
SB_DATA_PATH = pathlib.Path(settings.ROOT_PATH) / 'data' / 'science_birds'
SB_CONFIG_PATH = SB_DATA_PATH / 'config'
NOVELTY_LEVELS = {'1': 'novelty_level_1', '2': 'novelty_level_2', '3': 'novelty_level_3'}
NOVELTY_TYPES = {"6": "type6", "7": "type7", "8": "type8", "9": "type9", "10": "type10"}

TRAINING_SETS = [
    os.path.join('M18', '100_level_1_type_6_novelties.xml'),
    os.path.join('M18', '100_level_1_type_7_novelties.xml'),
    os.path.join('M18', '100_level_1_type_8_novelties.xml'),
    os.path.join('M18', '100_level_1_type_9_novelties.xml'),
    os.path.join('M18', '100_level_1_type_10_novelties.xml'),
    os.path.join('M18', '100_level_2_type_6_novelties.xml'),
    os.path.join('M18', '100_level_2_type_7_novelties.xml'),
    os.path.join('M18', '100_level_2_type_8_novelties.xml'),
    os.path.join('M18', '100_level_2_type_9_novelties.xml'),
    os.path.join('M18', '100_level_2_type_10_novelties.xml'),
    os.path.join('M18', '100_level_3_type_6_novelties.xml'),
    os.path.join('M18', '100_level_3_type_7_novelties.xml')
]

"""
Creates an .xml config file that contains levels not found in a particular dataset (ie, another (set) of .xml config files)

NOTE: For now, will simply print out the filenames to be used
"""

# TEMP
PREFIX = './Levels/novelty_level_1/type6/Levels'

CONFIG_SETTINGS = '<?xml version="1.0" encoding="utf-16"?> \
<evaluation> \
  <novelty_detection_measurement step="1" measure_in_training="True" measure_in_testing="True" /> \
  <trials> \
     <trial id="0" number_of_executions="1" checkpoint_time_limit="200" checkpoint_interaction_limit="200" notify_novelty="False"> \
      <game_level_set mode="training" time_limit="700000" total_interaction_limit="500000" attempt_limit_per_level="1" allow_level_selection="False">'

if __name__ == "__main__":
    training_set = set()

    # Collect levels found in training sets
    for training in TRAINING_SETS:
        training_path = os.path.join(SB_CONFIG_PATH, training)
        # print("Parsing: {}".format(training_path))

        tree = ET.parse(training_path)

        root = tree.getroot()

        for node in root:
            # print("ITERATE OVER NODE {}".format(node.tag))
            if node.tag == 'trials':
                # Go two levels down to access children of game_level_set
                levels = node[0][0]
                for level in levels:
                    lp = level.attrib['level_path']
                    # print("level path is: {}".format(lp))
                    lp = lp.split(PREFIX)[0]
                    training_set.add(lp)


    # Sample levels randomly from overall level pool (NOTE: Possible that if LEVELS_PER_NOVELTY is high enough, we will not get enough and cause an error)
    for novelty, types in NOVELTIES.items():
        for ntype in types:
            levels = set()

            pool = os.listdir(os.path.join(SB_LEVEL_PATH, NOVELTY_LEVELS[novelty], NOVELTY_TYPES[ntype], 'Levels'))

            for _ in range(10000):
                if len(levels) > LEVELS_PER_NOVELTY:
                    break
                else:
                    a_level = random.sample(pool, 1)[0]
                    if a_level in training_set:
                        continue
                    else:
                        levels.add(a_level)

            with open("level_{}_type_{}.txt".format(novelty, ntype), 'w+') as f:
                for level in levels:
                    f.write('<game_levels level_path = "./Levels/{}/{}/Levels/{}" />\n'.format(NOVELTY_LEVELS[novelty], NOVELTY_TYPES[ntype], level))


    # Add to xml

    # Write to file 
    # print("Writing trials to JSON file: {}".format(TRIALS_FILENAME))
    # with open(TRIALS_FILENAME, "w+") as f:
    #     json.dump(trials, f, sort_keys=True, indent=4)
