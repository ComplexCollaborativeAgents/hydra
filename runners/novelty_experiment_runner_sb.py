import csv
import datetime
import json
import logging
import os
import pathlib
from typing import List
from xml.etree import ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import pandas
import seaborn as sns
from pandas.core.frame import DataFrame

import settings
from agent.sb_hydra_agent import RepairingSBHydraAgent, SBHydraAgent
from dispatcher.sb_dispatcher import SBDispatcher
from runners.constants import KNOWN, NON_NOVELTY_PERFORMANCE, NOVELTY, UNKNOWN
from runners.run_sb_stats import (AgentType, diff_directories,
                                  glob_directories, prepare_config)
from utils.generate_trials_sb import (collect_levels, generate_trial_sets,
                                      unpack_trial_id)
from utils.stats import (AgentStats, NoveltyDetectionStats, SBAgentStats,
                         SBDetectionStats)

# from runners.run_sb_stats import *

logging.basicConfig(format='%(name)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("novelty_experiment_runner")
logger.setLevel(logging.INFO)

# Paths
SB_BIN_PATH = pathlib.Path(settings.SCIENCE_BIRDS_BIN_DIR) / 'linux'
SB_DATA_PATH = pathlib.Path(settings.ROOT_PATH) / 'data' / 'science_birds'
SB_CONFIG_PATH = SB_DATA_PATH / 'config'
TEMPLATE_PATH = SB_CONFIG_PATH / 'test_config.xml'

# Options
RESULTS_PATH = pathlib.Path(settings.ROOT_PATH) / "runners" / "experiments" / "ScienceBirds" / "SB_experiment"
EXPORT_TRIALS = False   # Export trials xml file
NUM_TRIALS = 1      # Number of trials to generate
NUM_LEVELS = 30    # Levels per trial
LEVELS_BEFORE_NOVELTY = 7  # Levels before novelty is introduced
NOTIFY_NOVELTY = False
REPETITION = 1   # If not set to None, the same sampled level will be used this many times before another is selected.
NON_NOVEL_TO_USE = { # level and type of non-novel levels to use
    'novelty_level_0': [
        "type1230"
        #"type2243"
        #"type2242"
        #"type2241"
        #"type2240"
        #"type2250"
        #"type2251"
        #"type2252"
        #"type2253"
        #"type2254"
        #"type2342"
        #"type2373"
        #"type2461"
        #"type2462"
        #"type2463"
        #"type2464"
        #"type2470"
        #"type2471"
        #"type2472"
        #"type2473"
        #"type2474"
        #"type2560"
        #"type2561"
        #"type2562"
        #"type2563"
        #"type2570"
        #"type2571"
        #"type2572"
        #"type2573"
        #"type2574"
        ]
    }
NOVEL_TO_USE = {    # level and type of novel levels to use
    'novelty_level_12': [
        'type30'
    ],
    # 'novelty_level_22': [
    #     'type43'
    # ],
    # 'novelty_level_37':[
    #     'type20'
    # ],
    # 'novelty_level_38':[
    #     'type20',
    #     'type10',
    #     'type11',
    #     'type12',
    #     'type13',
    #     'type14'
    # ]
}


def load_lookup(lookup_path):
    with open(lookup_path) as f:
        obj = json.load(f)
        return obj


class NoveltyExperimentRunnerSB:
    """ Runs experiments for science birds using M18 metrics """

    def __init__(self,
                 agent_type: AgentType,
                 novelties: dict = NOVEL_TO_USE,
                 non_novelties: dict = NON_NOVEL_TO_USE,
                 num_trials: int = NUM_TRIALS,
                 num_levels: int = NUM_LEVELS,
                 levels_before_novelty: int = LEVELS_BEFORE_NOVELTY,
                 repetition: bool = REPETITION,
                 export_trials: bool = EXPORT_TRIALS,
                 results_path: pathlib.Path = RESULTS_PATH):
        
        self.novelties = novelties
        self.non_novelties = non_novelties
        self.num_trials = num_trials
        self.repetition = repetition
        self.num_levels = num_levels
        self.levels_before_novelty = levels_before_novelty
        self.export_trials = export_trials

        if agent_type == AgentType.Hydra:
            self.agent = SBHydraAgent()
        elif agent_type == AgentType.RepairingHydra:
            self.agent = RepairingSBHydraAgent()
        else:
            self.agent = SBHydraAgent()
        
        self.dispatcher = SBDispatcher(self.agent)

        self._results_directory_path = results_path
        if not os.path.exists(self._results_directory_path):
            os.makedirs(self._results_directory_path)

        self.levels = dict()

        # Collect levels
        self.load_levels()

        # Setting up stats:
        self.data_categories = [
            'trial_id', 'trial_number', 'level', 'episode_number', 'performance',
            'trial_type', 'episode_type', 
            'novelty_level', 'novelty_type',
            'novelty_threshold'
        ]
        self.data_categories.extend(list(AgentStats.__annotations__.keys()))
        self.data_categories.extend(list(NoveltyDetectionStats.__annotations__.keys()))
        self.data_categories.extend(list(SBAgentStats.__annotations__.keys()))
        self.data_categories.extend(list(SBDetectionStats.__annotations__.keys()))
        

    def load_levels(self):
        """
        Collect levels to be drawn upon when constructing trials
        """

        self.levels = collect_levels()

    def reset(self):
        """ reset the experiment """
        self.levels.clear()
        self.load_levels()

    def prepare_config(self, config:str, level_list:List[str], notify_novelty:bool):
        """_summary_

        Args:
            config (str): Filepath to the config file that the science birds app will use
            trial (List[str]): List of filepaths to levels to use in the trial
            notify_novelty (bool): whether or not the trial is meant to inform the agent if there is novelty or not
        """

        # Create xml config file for the trial
        tree = ET.parse(TEMPLATE_PATH)

        if notify_novelty is not None:
            xpath = './trials/trial'
            trial = tree.getroot().find(xpath)
            trial.set('notify_novelty', str(notify_novelty))

        xpath = './trials/trial/game_level_set'
        level_set = tree.getroot().find(xpath)
        level_set.set('time_limit', '500000')
        level_set.set('total_interaction_limit', '1000000')

        for child in list(level_set):
            level_set.remove(child)

        for level in level_list:
            relpath = os.path.relpath(level, SB_BIN_PATH)
            ET.SubElement(level_set, 'game_levels', level_path=relpath)

        tree.write(config)

    def run_trial(self, trial_id:str, trial: List[str], trial_num: int, trial_type: str,  novelty_level: str, config_file: pathlib.Path = None) -> pandas.DataFrame:
        """ Run a trial """

        do_notify = trial_type == KNOWN

        if config_file is None:
            # Construct the config file
            config = SB_CONFIG_PATH / 'stats_config.xml'

            if self.export_trials:  # Export to unique trial xml config file
                date_time_str = datetime.datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
                config = SB_CONFIG_PATH / "trial_config_{}_{}.xml".format(trial_num, date_time_str)
                logger.debug("Exporting trial to {}".format(config))

            self.prepare_config(config, trial, do_notify)
        else:
            config = config_file

        # Run the agent
        self.dispatcher.set_trial_info(trial_id, config)    # Set info for dispatcher to use
        self.dispatcher.run()

        # Get agent stats
        agent_results_list = list(self.dispatcher.results.values())[0]

        results_directories = glob_directories(SB_BIN_PATH, 'Agent*')
        results_directories.sort(key=lambda x: str(x))
        results_directory = results_directories[-1]

        return self.compute_trial_stats(results_directory, agent_results_list, trial_num, trial_type, novelty_level)

    def compute_trial_stats(self,
                            results_directory: str,
                            agent_results_list: List[dict],
                            trial_num: int,
                            trial_type: str,
                            novelty_level: str) -> pandas.DataFrame:
        """
        Given a results directory, compute the statistics of the agent's performance.
        agent_stats is a list of dicts - stats of the agent that are not produced by the Angry Birds simulation
        """
        
        trial = pandas.DataFrame(columns=self.data_categories)

        # bird_scores = collections.defaultdict(lambda: {"passed": 0, "failed": 0})
        logger.debug("Gathering stats from: {}".format(results_directory))
        # Open results .csv
        evaluation_data = list(results_directory.glob('*_EvaluationData.csv'))
        for eval_data in evaluation_data:
            with open(eval_data) as f:
                data = csv.DictReader(f)
                for agent_results, row in zip(agent_results_list, data):
                    # birds = get_bird_count(os.path.join(SB_BIN_PATH, level_path))
                    status = row['LevelStatus']
                    level_name = row['levelName']

                    # Novelty type is always the 5th index (0 indexed)
                    novelty_type = level_name.split(os.path.sep)[5]
                    
                    is_novelty = NOVELTY

                    if row['novelty'] == 'basic':
                        is_novelty = NON_NOVELTY_PERFORMANCE

                    if 'Pass' not in status:
                        # for bird in birds.keys():
                        #     bird_scores[bird]['passed'] += 1
                        status = 'Fail'
                        # for bird in birds.keys():
                        #     bird_scores[bird]['failed'] += 1

                    score = float(row['Score']) 

                    novelty_threshold = 1   # TODO: figure out how to extract this

                    # Override novelty level value based on whether this is from novelty level 0 or not
                    nl = novelty_level if is_novelty == NOVELTY else "0"

                    data = {
                        'trial_type': [trial_type],
                        'novelty_level': [nl],
                        'novelty_type': [novelty_type],
                        'episode_type': [is_novelty],
                        'novelty_threshold': [novelty_threshold],
                        'performance': [score],
                    }

                    # Update data with experiment results
                    for key, value in agent_results.items():
                        if key in data and type(data[key]) == list:
                            data[key].append(value)
                        else:
                            data[key] = [value]  

                    # Override values with one gotten from results file
                    data['level'] = [level_name]
                    data['success'] = [status]

                    result = pandas.DataFrame(data=data)

                    trial = trial.append(result)

        return trial

    def run_experiment(self, configs: list = None):
        """ Run a set of trials defined by self.novelties OR by a set of config files """
        
        trial_num = 0
        experiment_results = pandas.DataFrame(columns=self.data_categories)
        
        experiment_results_path = os.path.join(self._results_directory_path, "novelty_results.csv")
        with open(experiment_results_path, "w+") as f:
            experiment_results.to_csv(f, index=False)

            if configs is None:
                # Generate a set of trials as according to above settings
                trial_sets = generate_trial_sets(self.num_trials, self.num_levels, self.levels_before_novelty, 
                                                    self.repetition, self.non_novelties, self.novelties, self.levels)
                
                #for trial_type in [UNKNOWN, KNOWN]:
                for trial_type in [UNKNOWN]:
                    for trial_id, trial in trial_sets.items():
                        # Extract novelty level from the trial id
                        _, _, novelty_level, _, _  = unpack_trial_id(trial_id)
                        logger.info("Starting trial {}".format(trial_id))

                        # Run the trial
                        trial_results = self.run_trial(trial_id, trial, trial_num, trial_type, novelty_level)
                        experiment_results = experiment_results.append(trial_results)   # Update dataframe

                        # Export results to file
                        with open(experiment_results_path, "a") as f:
                            trial_results.to_csv(f, index=False, header=False)

                        trial_num += 1

            else:   # Configs is not None, use that instead of generating a new trial
                #for trial_type in [UNKNOWN, KNOWN]:
                for trial_type in [UNKNOWN]:
                    for _ in range(self.num_trials):
                        for config in configs:
                            logger.debug("Using config file: {}".format)

                            # Get trial details
                            tree = ET.parse(config)

                            if tree.getroot().find('./trials/trial').get('notify_novelty'):
                                notify_novelty = KNOWN
                            else:
                                notify_novelty = UNKNOWN
                            novelty_level = "0"

                            level_tags = tree.getroot().find('./trials/trial/game_level_set')
                            for level_tag in level_tags:
                                level_path = level_tag.attrib['level_path']
                                if 'novelty_level_0' in level_path:
                                    continue
                                else:
                                    novelty_level = level_path.split('/')[2]    # TODO: make this cleaner

                            # Run the trial
                            trial_results = self.run_trial(config, [], trial_num, trial_type, novelty_level, config_file=config)
                            experiment_results = experiment_results.append(trial_results)

                            # Export results to file
                            with open(experiment_results_path, "a") as f:
                                trial_results.to_csv(f, index=False, header=False)

                            trial_num += 1

    @staticmethod
    def categorize_examples_for_novelty_detection(dataframe):
        dataframe['predicted_novel'] = np.where(dataframe['novelty_probability'] < dataframe['novelty_threshold'], False, True)
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

    @staticmethod
    def get_program_metrics(cdt: pandas.DataFrame, trials: pandas.DataFrame, hydra: pandas.DataFrame, baseline: pandas.DataFrame, asymptote: int = 5):
        num_trials_per_type = trials.groupby("trial_type").agg({'FN': len}).rename(columns={'FN': 'count'}) # Get number of trials
        # Aggregate to get
        scores = cdt.groupby("trial_type").agg({'FN': np.mean, 'FP': len}).rename(columns={'FN': 'M1', 'FP': 'CDT_count'})

        # M1 - average number of FN among CDTs

        # M2 - % of CDTs
        scores['M2'] = scores['CDT_count'] / num_trials_per_type['count']

        # M2.1 - % of trials with at least 1 FP
        trials['CDT_FP_count'] = np.where((trials['FP'] > 0), True, False)

        CDT_FP_count = trials[trials['CDT_FP_count'] == True].groupby("trial_type")
        CDT_FP_count = CDT_FP_count.agg({'CDT_FP_count': len})

        scores['M2.1'] = CDT_FP_count.replace(np.nan, 0)
        scores['M2.1'] = scores['M2.1'] / num_trials_per_type['count']

        performance_metrics = pandas.DataFrame(columns=['M3', 'M4', 'M5', 'M6'])
        grouped_trials = hydra[['trial_num', 'trial_type', 'episode_type', 'performance']].groupby("trial_num")
        grouped_base = baseline[['trial_num', 'trial_type', 'episode_type', 'performance']].groupby("trial_num")

        m3_agg_trials = pandas.DataFrame(columns=["index", "0", 'known_perf'])
        m4_agg_trials = pandas.DataFrame(columns=["index", "0", 'unknown_perf'])
        m5_agg_trials = pandas.DataFrame(columns=["episode_type", "performance", "trial_num", "trial_type"])
        m6_agg_trials = pandas.DataFrame(columns=["episode_type", "performance", "trial_num", "trial_type"])
        m3_agg_base = pandas.DataFrame(columns=["index", "0", 'known_perf'])
        m4_agg_base = pandas.DataFrame(columns=["index", "0", 'unknown_perf'])
        m5_agg_base = pandas.DataFrame(columns=["episode_type", "performance", "trial_num", "trial_type"])
        m6_agg_base = pandas.DataFrame(columns=["episode_type", "performance", "trial_num", "trial_type"])

        trials_known_avg = []   # M3
        trials_unknown_avg = [] # M4
        trials_last = []        # M5
        trials_asymptote = []   # M6
        for trial_num, group in grouped_trials:
            if 'known' in group.values: # group.trial_type: # M3
                known_avg = group.groupby(["trial_type", "episode_type"]).get_group(("known", "novelty")).agg({'performance': np.mean})
                trials_known_avg.append(known_avg)

            if 'unknown' in group.values: # group.trial_type: # M4
                unknown_avg = group.groupby(["trial_type", "episode_type"]).get_group(("unknown", "novelty")).agg({'performance': np.mean})
                trials_unknown_avg.append(unknown_avg)

            # M5
            last_per_group = group.groupby("episode_type").get_group("novelty").agg({'performance': np.sum})
            trials_last.append(last_per_group)

            # M6
            novelty_grouped = group.groupby("episode_type").get_group("novelty")
            slice = asymptote if len(novelty_grouped.index) > asymptote else len(novelty_grouped.index)
            post_asymptote = novelty_grouped.tail(slice).agg({"performance": np.sum})
            trials_asymptote.append(post_asymptote)

        if len(trials_known_avg) > 0:
            m3_agg_trials = pandas.concat(trials_known_avg, sort=True).reset_index()
        if len(trials_unknown_avg) > 0:
            m4_agg_trials = pandas.concat(trials_unknown_avg, sort=True).reset_index()
        if len(trials_last) > 0:
            m5_agg_trials = pandas.concat(trials_last, sort=True).reset_index()
        if len(trials_asymptote) > 0:
            m6_agg_trials = pandas.concat(trials_asymptote, sort=True).reset_index()

        base_known_avg = []     # M3
        base_unknown_avg = []   # M4
        base_last = []          # M5
        base_asymptote = []     # M6
        for trial_num, group in grouped_base:
            if 'known' in group.values: # group.trial_type:    # M3
                known_avg = group.groupby(["trial_type", "episode_type"]).get_group(("known", "non-novelty-performance")).agg({'performance': np.mean})
                base_known_avg.append(known_avg)

            if 'unknown' in group.values: # group.trial_type:    # M4
                unknown_avg = group.groupby(["trial_type", "episode_type"]).get_group(("unknown", "non-novelty-performance")).agg({'performance': np.mean})
                base_unknown_avg.append(unknown_avg)

            # M5
            last_per_group = group.groupby("episode_type").get_group("novelty").agg({'performance': np.sum})
            base_last.append(last_per_group)

            # M6
            novelty_grouped = group.groupby("episode_type").get_group("novelty")
            slice = asymptote if len(novelty_grouped.index) > asymptote else len(novelty_grouped.index)
            post_asymptote = novelty_grouped.tail(slice).agg({"performance": np.sum})
            base_asymptote.append(post_asymptote)

        if len(base_known_avg) > 0:
            m3_agg_base = pandas.concat(base_known_avg, sort=True).reset_index()
        if len(base_unknown_avg) > 0:
            m4_agg_base = pandas.concat(base_unknown_avg, sort=True).reset_index()
        if len(base_last) > 0:
            m5_agg_base = pandas.concat(base_last, sort=True).reset_index()
        if len(base_asymptote) > 0:
            m6_agg_base = pandas.concat(base_asymptote, sort=True).reset_index()

        # print("m3 trial: {} m3 base: {}".format(m3_agg_trials.empty, m3_agg_base.empty))
        # print("m4 trial: {} m4 base: {}".format(m4_agg_trials.empty, m4_agg_base.empty))
        # print("m5 trial: {} m5 base: {}".format(m5_agg_trials.empty, m5_agg_base.empty))
        # print("m6 trial: {} m6 base: {}".format(m6_agg_trials.empty, m6_agg_base.empty))

        if not m3_agg_trials.empty and not m3_agg_base.empty:
            m3_agg_trials['known_perf'] = m3_agg_trials[0] / m3_agg_base[0]
            performance_metrics['M3'] = m3_agg_trials.agg({'known_perf': np.mean})
        if not m4_agg_trials.empty and not m4_agg_trials.empty:
            m4_agg_trials['unknown_perf'] = m4_agg_trials[0] / m4_agg_base[0]
            performance_metrics['M4'] = m4_agg_trials.agg({'unknown_perf': np.mean})['unknown_perf']
        if not m5_agg_trials.empty and not m5_agg_base.empty:
            m5_agg_trials['opti_t'] = m5_agg_trials[0] / (m5_agg_base[0] + m5_agg_trials[0])
            performance_metrics['M5'] = m5_agg_trials.agg({'opti_t': np.sum})[0] / len(m5_agg_trials.index)
        if not m6_agg_trials.empty and not m6_agg_base.empty:
            m6_agg_trials['asymptote'] = m6_agg_trials[0] / m6_agg_base[0].replace({0: np.nan})
            performance_metrics['M6'] = m6_agg_trials.agg({'asymptote': lambda x: x.sum(skipna=True)})['asymptote'] / len(m6_agg_trials.index)

        return scores, performance_metrics

    @staticmethod
    def plot_experiment_results(df, novelty_episode_number):
        plt.figure(figsize=(16, 9))
        ax = sns.lineplot(data=df, y='performance', x='episode_num', hue='trial_type', ci=95)
        ax.set(ylim=(0, 150000))
        plt.axvline(x=novelty_episode_number, color='red')
        plt.title("Experiment results", fontsize=20)
        plt.xlabel("episodes", fontsize=15)
        plt.ylabel("performance", fontsize=15)


if __name__ == '__main__':
    experiment_runner = NoveltyExperimentRunnerSB(AgentType.RepairingHydra, export_trials=True)
    experiment_runner.run_experiment()
    # experiment_runner.run_experiment(configs=[SB_CONFIG_PATH / "debug_config.xml"])
