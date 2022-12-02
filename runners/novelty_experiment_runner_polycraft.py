import json
import os
import logging
import pathlib
import pandas
import numpy as np

from typing import List

import settings
from agent.polycraft_hydra_agent import PolycraftHydraAgent
from utils.generate_trials_polycraft import (collect_levels,
                                             generate_trial_sets,
                                             unpack_trial_id)
from worlds.polycraft_world import Polycraft, ServerMode

from runners.constants import KNOWN, NON_NOVELTY_PERFORMANCE, NOVELTY, UNKNOWN
from runners.polycraft_dispatcher import PolycraftDispatcher

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
NUM_LEVELS = 5     # Levels per trial
LEVELS_BEFORE_NOVELTY = 3   # Levels before novelty is introduced
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

        # NOTE: As of 6/16/2022, The only working setup is to run in Tournament mode
        self.agent = PolycraftHydraAgent()
        self.dispatcher = PolycraftDispatcher(self.agent)

        self.levels = collect_levels()

        # Setup trials
        self.trials = {}
        self.setup_trials(save_trials=True)

        # Results of a trial
        self.results = {}

    def setup_trials(self, save_trials:bool=False):
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
        self.dispatcher.experiment_start(standalone=True)
        self.dispatcher.set_trial_sets(self.trials)
        self.dispatcher.run_trials()

        self.process_experiment()

        self.dispatcher.experiment_end()

    def run_trial_from_json(self, json_path:str):
        """ Run a single trial from a json file output by the novelty experiment runner

        Args:
            json_path (str): path to the json file.
        """
        with open(json_path, 'r') as f:
            trial = json.load(f)

            self.dispatcher.experiment_start(standalone=True)
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
            
            df = pandas.DataFrame(data={
                'trial_num': [trial_num],               # Trial number
                'trial_type': [UNKNOWN],                 # Whether or not agent is supplied with novelty presence or not    # NOTE Currently all supplied novelties are unknown
                'novelty_level': [novelty],             # Novelty level
                'novelty_type': [n_type],               # Novelty type
                'novelty_subtype': [n_stype],           # Novelty subtype
                'episode_type': [is_novelty],           # Ground truth of novelty existence
                'episode_num': [episode_num],           # Episode number
                'novelty_probability': [results['novelty_likelihood']],        # Agent supplied novelty probability
                'novelty_threshold': ['None'],          # Agent supplied novelty threshold (NOTE in sb runner, used in comparison to say whether or not novelty was detected)
                'novelty': ['None'],                    # Not used
                'novelty_characterization': ['None'],   # Agent supplied description of novelty
                'predicted_novel': [results['novelty']], # Agent detected novelty
                'performance': [results['step_cost']],  # Step cost of the agent
                'pass': [results['passed']],            # Whether the agent passed or not
                'num_repairs': [results['repair_calls']],                # Agent supplied number of repairs done for this episode
                'repair_time': [results['repair_time']]                 # Agent supplied time it took to repair for this episode
            })

            trial = trial.append(df)

        return trial

    def process_experiment(self):
        experiment_df = pandas.DataFrame(columns=['trial_num', 'trial_type', 'novelty_level', 'novelty_type', 'novelty_subtype', 'episode_type',
                                                  'episode_num', 'novelty_probability',
                                                  'novelty_threshold', 'novelty', 'novelty_characterization', 'predicted_novel',
                                                  'performance', 'pass', 'num_repairs', 'repair_time', 'pddl_novelty_likelihood', 'unknown_object', 'reward_estimator_likelihood'])

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
        dataframe['TN'] = np.where((dataframe['episode_type'] == NON_NOVELTY_PERFORMANCE) & (dataframe['predicted_novel'] == False), 1, 0)
        dataframe['FP'] = np.where((dataframe['episode_type'] == NON_NOVELTY_PERFORMANCE) & (dataframe['predicted_novel'] == True), 1, 0)
        dataframe['TP'] = np.where((dataframe['episode_type'] == NOVELTY) & (dataframe['predicted_novel'] == True), 1, 0)
        dataframe['FN'] = np.where((dataframe['episode_type'] == NOVELTY) & (dataframe['predicted_novel'] == False), 1, 0)
        return dataframe

    @staticmethod
    def get_trials_summary(dataframe):
        # print(dataframe)
        trials = dataframe[['trial_num', 'trial_type', 'pass', 'FN', 'FP', 'TN', 'TP', 'performance']]
        trials['passed'] = np.where(trials['pass'] == 'Pass', 1, 0)
        # print(trials)

        grouped = trials.groupby(['trial_type', 'trial_num'])

        # print("AFTER group output: \n{}".format(grouped))

        aggregate = grouped.agg({'FN': np.sum, 'FP': np.sum, 'TN': np.sum, 'TP': np.sum, 'performance': np.mean, 'passed': np.sum})
        
        # print("AFTER aggregate output: \n{}".format(aggregate))

        aggregate['is_CDT'] = np.where((aggregate['TP'] > 1) & (aggregate['FP'] == 0), True, False)
        
        # print("AFTER CDT: {}".format(trials))
        cdt = aggregate[aggregate['is_CDT'] == True]

        return aggregate, cdt

if __name__ == "__main__":
    runner = NoveltyExperimentRunnerPolycraft()
    runner.run_trials()
    # runner.run_trial_from_json("0_3_POGO_L01_T01_S01_E.json")  # Insert path to json file here! (Do not commit any changes to this)