import json
import os
import logging
import pathlib
import pandas
import dataclasses
import numpy as np

from typing import List, Dict

import settings
from agent.polycraft_hydra_agent import PolycraftHydraAgent
from utils.generate_trials_polycraft import (collect_levels,
                                             generate_trial_sets,
                                             unpack_trial_id)
from utils.stats import AgentStats, NoveltyDetectionStats, PolycraftAgentStats, PolycraftDetectionStats
from worlds.polycraft_world import Polycraft, ServerMode

from runners.constants import KNOWN, NON_NOVELTY_PERFORMANCE, NOVELTY, UNKNOWN
from dispatcher.polycraft_dispatcher import PolycraftDispatcher

"""
Runs a 
"""

# Logger
logging.basicConfig(format='%(name)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("novelty_experiment_runner")
logger.setLevel(logging.INFO)

# Settings
RESULTS_PATH = pathlib.Path(settings.ROOT_PATH) / "runners" / "experiments" / "Polycraft"
EXPERIMENT_NAME = 'novelty_results'
NUM_TRIALS = 1      # Number of trials to generate
NUM_LEVELS = 3     # Levels per trial
LEVELS_BEFORE_NOVELTY = 1   # Levels before novelty is introduced
REPETITION = 1   # If not set to None, the same sampled level will be used this many times before another is selected.
DIFFICULTIES = ['E'] # E = Easy, M = Medium, H = Hard, A = All/Mixed difficulty
NON_NOVEL_TO_USE = { # level and type of non-novel levels to use (The ones listed here are all that are currently available)
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

class NoveltyExperimentRunnerPolycraft:


    def __init__(self):
        """"""
        agent: PolycraftHydraAgent
        dispatcher: PolycraftDispatcher
        levels: Dict[str, List[str]]
        trials: Dict[str, List[str]]
        data_categories: List[str]

        # NOTE: As of 6/16/2022, The only working setup is to run in Tournament mode
        self.agent = PolycraftHydraAgent()
        self.dispatcher = PolycraftDispatcher(self.agent)

        self.levels = collect_levels()

        # Setup trials
        self.trials = {}
        self.setup_trials(save_trials=True)

        # Results of a trial
        self.results = {}

        # Captured statistics in results
        self.data_categories = [
            'trial_id', 'trial_number', 'level', 'episode_number', 'step_cost',
            'trial_type', 'episode_type', 
            'novelty_level', 'novelty_type', 'novelty_subtype',
            'novelty_threshold'
        ]
        self.data_categories.extend(list(AgentStats.__annotations__.keys()))
        self.data_categories.extend(list(NoveltyDetectionStats.__annotations__.keys()))
        self.data_categories.extend(list(PolycraftAgentStats.__annotations__.keys()))
        # self.data_categories.extend(list(PolycraftDetectionStats.__annotations__.keys()))

    def setup_trials(self, save_trials:bool=False):
        """ From the set of collected levels, generate trials using the settings

        Args:
            save_trials (bool, optional): _description_. Defaults to False.
        """

        self.trials = {}
        self.trials = generate_trial_sets(NUM_TRIALS, NUM_LEVELS, LEVELS_BEFORE_NOVELTY, REPETITION,
                                          NON_NOVEL_TO_USE, NOVEL_TO_USE, DIFFICULTIES, self.levels)

        # Save trial
        if save_trials:
            for trial_id, trial in self.trials.items():
                with open("{}.json".format(trial_id), 'w+') as f:
                    json.dump(trial, f, indent=4)

    # Run trials
    def run_trials(self):
        """_summary_
        """

        self.dispatcher.run_experiment(standalone=True)
        self.dispatcher.set_trial_sets(self.trials)
        self.dispatcher.run_trials()

        self.process_experiment()

        self.dispatcher.cleanup_experiment()

    def run_trial_from_json(self, json_path:str):
        """ Run a single trial from a json file output by the novelty experiment runner

        Args:
            json_path (str): path to the json file.
        """
        with open(json_path, 'r') as f:
            trial = json.load(f)

            self.dispatcher.set_trial_sets({
                json_path: trial
            })
            self.dispatcher.run_trials()

            self.process_experiment()
            self.dispatcher.experiment_end()

    def process_trial_results(self, trial_id:str, results_list:List):
        trial = pandas.DataFrame(columns=['trial_num', 'trial_type', 'novelty_level', 'novelty_type', 'novelty_subtype',
                                              'episode_type', 'episode_num', 'novelty_probability',
                                              'novelty_threshold', 'novelty', 'novelty_characterization',
                                              'predicted_novel', 'performance', 'pass', 'num_repairs', 'repair_time'])

        trial_num, num_levels, novelty, n_type, n_stype, difficulty = unpack_trial_id(trial_id)
        
        for episode_num, results in enumerate(results_list):
            
            is_novelty = NON_NOVELTY_PERFORMANCE if episode_num < LEVELS_BEFORE_NOVELTY else NOVELTY
            
            data = {
                'trial_type': [UNKNOWN],                # Whether or not agent is supplied with novelty presence or not    # NOTE Currently all supplied novelties are unknown
                'episode_type': [is_novelty],           # Ground truth of novelty existence
                'novelty_level': [novelty],             # Novelty level
                'novelty_type': [n_type],               # Novelty type
                'novelty_subtype': [n_stype],           # Novelty subtype
                'novelty_threshold': ['None'],          # Agent supplied novelty threshold (NOTE in sb runner, used in comparison to say whether or not novelty was detected)
            }           

            # Update results with experiment metadata
            for key, value in results.items():
                if key in data and type(data[key]) == list:
                    data[key].append(value)
                else:
                    data[key] = [value]  

            df = pandas.DataFrame(data=data)

            trial = trial.append(df)

        print(trial)
        return trial

    def process_experiment(self):
        experiment_df = pandas.DataFrame(columns=self.data_categories)

        experiment_results_path = os.path.join(RESULTS_PATH, "{}.csv".format(EXPERIMENT_NAME))
        with open(experiment_results_path, "w+") as f:
            experiment_df.to_csv(f, index=False)

        for trial_id, results_list in self.dispatcher.results.items():
            trial_df = self.process_trial_results(trial_id, results_list)

            experiment_df = experiment_df.append(trial_df)
        
            # Export results to file
            with open(experiment_results_path, "a") as f:
                trial_df.to_csv(f, index=False, header=False)

    @staticmethod
    def categorize_examples_for_predicted_novel(dataframe):
        dataframe['TN'] = np.where((dataframe['episode_type'] == NON_NOVELTY_PERFORMANCE) & (dataframe['novelty_detection'] == False), 1, 0)
        dataframe['FP'] = np.where((dataframe['episode_type'] == NON_NOVELTY_PERFORMANCE) & (dataframe['novelty_detection'] == True), 1, 0)
        dataframe['TP'] = np.where((dataframe['episode_type'] == NOVELTY) & (dataframe['novelty_detection'] == True), 1, 0)
        dataframe['FN'] = np.where((dataframe['episode_type'] == NOVELTY) & (dataframe['novelty_detection'] == False), 1, 0)
        return dataframe

    @staticmethod
    def get_trials_summary(dataframe):
        trials = dataframe[['trial_number', 'trial_type', 'success', 'FN', 'FP', 'TN', 'TP', 'step_cost']]
        trials['passed'] = np.where(trials['success'], 1, 0)

        grouped = trials.groupby(['trial_type', 'trial_number'])

        # print("AFTER group output: \n{}".format(grouped))

        aggregate = grouped.agg({'FN': np.sum, 'FP': np.sum, 'TN': np.sum, 'TP': np.sum, 'step_cost': np.mean, 'success': np.sum})
        
        # print("AFTER aggregate output: \n{}".format(aggregate))

        aggregate['is_CDT'] = np.where((aggregate['TP'] > 1) & (aggregate['FP'] == 0), True, False)
        
        # print("AFTER CDT: {}".format(trials))
        cdt = aggregate[aggregate['is_CDT'] == True]

        return aggregate, cdt

if __name__ == "__main__":
    runner = NoveltyExperimentRunnerPolycraft()
    runner.run_trials()
    # runner.run_trial_from_json()  # Insert path to json file here! (Do not commit any changes to this)
