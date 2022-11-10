import json
import logging
import os
import pathlib
import random
from enum import Enum
from typing import Dict, List, Tuple

import settings

logging.basicConfig(format='%(name)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("generate_trials")
logger.setLevel(logging.INFO)

class PolyDifficulty(Enum):
    EASY = 'E'
    MEDIUM = 'M'
    HARD = 'H'
    ALL = 'A'

# Settings
NUM_TRIALS = 2      # Number of trials to generate
NUM_LEVELS = 40     # Levels per trial
LEVELS_BEFORE_NOVELTY = 20   # Levels before novelty is introduced
REPETITION = 10   # If not set to None, the same sampled level will be used this many times before another is selected.   
OUTPUT_DIR = settings.ROOT_PATH
DIFFICUTIES = ['E'] # E = Easy, M = Medium, H = Hard, A = All/Mixed difficulty
NON_NOVEL_TO_USE = { # level and type of non-novel levels to use
    'L00': {
        'T01':[
            'S01'
        ]
    }
}   
NOVEL_TO_USE = {    # level and type of novel levels to use
    'L01': {
        'T01':[
            'S01'
        ]
    }
}

def format_trial_id(trial_num: int, num_levels: int, novelty: str, n_type: str, n_stype:str, difficulty:str, nn_type: str) -> str:
    """
    Format trial parameters into a semi-unique identifier
    """
    return '{}_{}_POGO_{}_{}_{}_{}'.format(trial_num, num_levels, novelty, n_type, n_stype, difficulty)


def unpack_trial_id(trial_id: str) -> Tuple:
    """
    Unpack a trial_id to get the parameters of the trial
    :returns: Tuple of size 5:
        1) trial_num: The trial number
        2) num_levels: Number of levels per trial
        3) novelty: Novelty level used in the trial
        4) n_type: Novelty type used in the trial
        5) n_stype: Novelty subtype used in the trial
        6) difficulty: Difficulty level of the trial (E, M, H)
    """
    segments = trial_id.split('_')

    trial_num = int(segments[0])
    num_levels = int(segments[1])
    novelty = segments[2]
    n_type = segments[3]
    n_stype = segments[4]
    difficulty = segments[5]

    return trial_num, num_levels, novelty, n_type, n_stype, difficulty

def unpack_trial_name(trial_name: str) -> Tuple:
    """
    Unpacks the values in the (official POGO) trial name
    """

    tokens = trial_name.split('_')
    try:
        novelty_level = tokens[1]       # LXX
        novelty_type = tokens[2]        # TXX
        novelty_subtype = tokens[3]     # SXX
        difficulty = tokens[5]          # E = Easy, M = Medium, H = Hard, A = All/Mixed difficulty
        notified = tokens[6][0] == "K"  # U means that the agent will not be informed of novelty presence. K means that novelty presence will be given.
        intro_at = int(tokens[6][1:])   # The number that follows is the episode number in which novelty is introduced.  The count starts at 0. 
        variant = tokens[7]

    except IndexError:
        raise ValueError("Invalid trial name: {}".format(trial_name))

    return novelty_level, novelty_type, novelty_subtype, difficulty, notified, intro_at, variant

def collect_levels() -> Dict[str, List[str]]:
    """
    Collects levels from the polycraft repository
    """
    all_levels = {}

    # Non-novelty
    pattern = "*.json"
    # logger.info("Adding non-novel levels from using pattern: {}".format(pattern))
    all_levels['L00'] = {}
    all_levels['L00']['T01'] = {}
    all_levels['L00']['T01']['S01'] = {}
    all_levels['L00']['T01']['S01']['E'] = [str(l) for l in pathlib.Path(settings.POLYCRAFT_NON_NOVELTY_LEVEL_DIR).glob(pattern)]

    # Novelty
    novelty_path = pathlib.Path(settings.POLYCRAFT_NOVELTY_LEVEL_DIR)
    # Process all novelty levels, types and subtypes
    for ndir in os.listdir(novelty_path):
        _, novelty_level, novelty_type, novelty_subtype = ndir.split('_')
        
        if novelty_level not in all_levels:
            all_levels[novelty_level] = {}

        if novelty_type not in all_levels[novelty_level]:
            all_levels[novelty_level][novelty_type] = {}

        if novelty_subtype not in all_levels[novelty_level][novelty_type]:
            all_levels[novelty_level][novelty_type][novelty_subtype] = {}

        for difficulty in ['E', 'M' 'H']:
            path = "{}/{}/X0100/{}_X0100_{}_U0015_V0".format(settings.POLYCRAFT_NOVELTY_LEVEL_DIR, ndir, ndir, difficulty)
            logger.info("Adding novel levels from from: {}".format(path))
            levels = [str(l) for l in pathlib.Path(path).glob(pattern)]
            all_levels[novelty_level][novelty_type][novelty_subtype][difficulty] = levels

    return all_levels


def generate_trial_sets(num_trials:int, num_levels:int, 
                        levels_before_novelty:int, repetition:int, 
                        non_novel_to_use:dict, novel_to_use:dict,
                        difficulties: List[str],
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
        for nn_type in non_novel_to_use['L00']:
            for novelty, novel_types in novel_to_use.items():
                # Create a trial for every type
                for n_type, n_stypes in novel_types.items():
                    for n_stype in n_stypes:
                        for difficulty in difficulties:
                            trial_id = format_trial_id(trial_num, num_levels, novelty, n_type, n_stype, difficulty, nn_type)
                            
                            logger.info("Generating trial for {}".format(trial_id))
                            trials[trial_id] = generate_trial(nn_type, novelty, n_type, n_stype, difficulty,
                                                            num_levels, levels_before_novelty, 
                                                            repetition, level_sets)

    return trials
        
def generate_trial(nn_type: str, novelty: str, n_type: str, n_stype: str, difficulty: str,
                    num_levels: int, levels_before_novelty: int, repetition: int,
                    level_sets: dict) -> List[str]:
    """
    Constructs a single trial using the provided non-novelty levels/types and novelty levels/types.
    :param nn_type: Non-novelty type that the trial uses (UNUSED)
    :param novelty: Novelty level that the trial uses
    :param n_type: Novelty type that the trial uses
    :param n_stype: Novelty subtype that the trial uses
    :param difficulty: difficulty of the levels that the trial uses
    :param num_levels: Number of levels per trial
    :param levels_before_novelty: Number of levels in a trial before novelty is introduced
    :param repetition: Number of times a level will be repeated within a trial
    :param level_sets: Nested dictionary with key:value pairs - novelty_level:novelty_type:list_of_levels.  Contains all collected non-novel levels/types as well as novelty levels/types
    """
    trial = []
    
    to_add = level_sets['L00']['T01']['S01']['E'][0] # placeholder
    for level_num in range(num_levels):
        # Use sampled non-novel level from set of non-novel levels to use
        if level_num < levels_before_novelty:
            # Update sampled level to add 
            # (if repetition is enabled, only update when repetition count is reached)
            if repetition is None or (repetition is not None and level_num % repetition == 0):
                index = random.choice(list(range(len(level_sets['L00']['T01']['S01']['E']))))
                to_add = level_sets['L00']['T01']['S01']['E'].pop(index) 
        else:   # Use level sampled from novelty level / type
            # Update sampled level to add 
            # (if repetition is enabled, only update when repetition count is reached)
            if repetition is None or (repetition is not None and level_num % repetition == 0):
                index = random.choice(list(range(len(level_sets[novelty][n_type][n_stype][difficulty]))))
                to_add = level_sets[novelty][n_type][n_stype][difficulty].pop(index) 
        # finally append the level
        trial.append(to_add)

    return trial


def save_trial_to_file(trial_levels:list, output_filepath:str):
    """
    Saves a trial into a .json file
    :param trial_levels: list of levels in the trial 
    :param output_filepath: filepath of the output json file 
    """
    
    with open(output_filepath, 'w+') as f:
        json.dump(trial_levels, f, indent=4)


def generate_eval_trial_sets():
    levels = collect_levels()   # Collect levels

    # Generate the trial sets
    trials = generate_trial_sets(NUM_TRIALS, NUM_LEVELS, LEVELS_BEFORE_NOVELTY,
                                 REPETITION,
                                 NON_NOVEL_TO_USE, NOVEL_TO_USE, DIFFICUTIES,
                                 levels)

    # Save each trial into a file
    for trial_id, trial in trials.items(): 
        output_filepath = "{}/{}.json".format(OUTPUT_DIR, trial_id)
        save_trial_to_file(trial, output_filepath)

if __name__ == "__main__":
    generate_eval_trial_sets()

    