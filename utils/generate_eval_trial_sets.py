import json
import pathlib
import random
import os

import settings

NUM_TRIALS = 2
NUM_LEVELS = 40     # Levels per trial
LEVELS_BEFORE_NOVELTY = 20   # Levels before novelty is introduced
SB_BIN_PATH = pathlib.Path(settings.SCIENCE_BIRDS_BIN_DIR) / 'linux'
NON_NOVELTY_DIR = os.path.join('Levels', 'novelty_level_0')
NOVELTY_LEVELS = {'novelty_level_1': '1', 'novelty_level_2': '2', 'novelty_level_3': '3', 'novelty_level_22': '22', 'novelty_level_23': '23', 'novelty_level_24': '24', 'novelty_level_25': '25'}
NOVELTY_TYPES = {"type1": "1", "type2": "2", "type4": "4", "type5": "5", "type6": "6", "type7": "7", "type8": "8", "type9": "9", "type10": "10", "type22": "22", "type23": "23", "type24": "24", "type25": "25"}
TRIALS_FILENAME = "eval_sb_trials_b4_novelty_{}.json".format(LEVELS_BEFORE_NOVELTY)

"""
Creates a JSON file of trials that have each start with non-novelty levels and transitions into novelty levels
Intended for 2021 evaluation

Output file format:
[
    {   // Trial 1
        "<Novelty Level>":{
            "<Novelty Type>":[
                "level file 1",
                "level file 2",
                ...
            ]
        },
    },
    { // Trial 2
        ...
    },
    ...
]

"""


if __name__ == "__main__":
    trials = []
    non_novelty_levels = []

    novelty_path = ""

    non_novelty_type_dirs = [os.path.join(t, "Levels") for t in os.listdir(SB_BIN_PATH / NON_NOVELTY_DIR)]

    # Collect non novelty levels
    for type_dir in non_novelty_type_dirs:
        type_dir_path = SB_BIN_PATH / NON_NOVELTY_DIR / type_dir
        for nn_filename in os.listdir(type_dir_path):
            print("Added {} to non-novelty levels".format(nn_filename))
            non_novelty_levels.append(os.path.join(NON_NOVELTY_DIR, type_dir, nn_filename))

    # Create trials
    for trial_num in range(1, NUM_TRIALS):
        
        trial = {}

        # For every novelty level
        for novelty_level in NOVELTY_LEVELS:
            novelty_path = os.path.join("Levels", novelty_level)
            
            print("Creating trial set for {}".format(novelty_path))
            sets = {}

            # For a novelty type within a novelty level, create a set of trials
            for novelty_type in NOVELTY_TYPES:
                trial_levels = []   # list of levels
                print("Processing trial for type {}, {}".format(novelty_type, novelty_level))

                # Check to see if type exists within the level dir
                if novelty_type not in os.listdir(os.path.join(SB_BIN_PATH, novelty_path)):
                    print("Novelty type {} not present in {}".format(novelty_type, novelty_path))
                    continue

                dir_path = os.path.join(SB_BIN_PATH, novelty_path, novelty_type, 'Levels')
                novelty_type_levels = os.listdir(dir_path)

                # Populate the trial with levels
                for index in range(NUM_LEVELS):
                    next_level = ""
                    if index < LEVELS_BEFORE_NOVELTY:   # append non novelty
                        next_index = random.randrange(len(non_novelty_levels))
                        next_level = non_novelty_levels[next_index]    # Pop random level from non novelty level
                        print("Appending non-novelty level {}".format(next_level))
                    else:   # append novelty
                        next_index = random.randrange(len(novelty_type_levels))
                        next_level = os.path.join(novelty_path, novelty_type, 'Levels', novelty_type_levels[next_index])    # Pop random level from novelty type
                        print("Appending novelty level {}".format(next_level))

                    trial_levels.append(next_level)

                # add to set
                sets[NOVELTY_TYPES[novelty_type]] = trial_levels

            # Add set to trial
            trial[NOVELTY_LEVELS[novelty_level]] = sets

        trials.append(trial)

    # Write to file 
    print("Writing trials to JSON file: {}".format(TRIALS_FILENAME))
    with open(TRIALS_FILENAME, "w+") as f:
        json.dump(trials, f, sort_keys=True, indent=4)
