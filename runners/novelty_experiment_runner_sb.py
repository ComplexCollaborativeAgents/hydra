import optparse
import pandas
import datetime
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
import seaborn as sns
from runners import constants

from xml.etree.ElementTree import ElementTree
from collections import defaultdict
from agent.gym_hydra_agent import *
from runners.run_sb_stats import *

logging.basicConfig(format='%(name)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("novelty_experiment_runner")
logger.setLevel(logging.INFO)

# Paths
SB_BIN_PATH = pathlib.Path(settings.SCIENCE_BIRDS_BIN_DIR) / 'linux'
SB_DATA_PATH = pathlib.Path(settings.ROOT_PATH) / 'data' / 'science_birds'
SB_CONFIG_PATH = SB_DATA_PATH / 'config'
TEMPLATE_PATH = SB_CONFIG_PATH / 'test_config.xml'

# Constants
NOVELTY_LEVELS = {'0': 'novelty_level_0', '1': 'novelty_level_1', '2': 'novelty_level_2',  '3': 'novelty_level_3', '22': 'novelty_level_22', '23': 'novelty_level_23', '24': 'novelty_level_24', '25': 'novelty_level_25'}
NOVELTY_TYPES = {"1": "type1", "2": "type2", "6": "type6", "7": "type7", "8": "type8", "9": "type9", "10": "type10"}
NON_NOVEL_LEVELS = ["0"]

# Options
RESULTS_PATH = pathlib.Path(settings.ROOT_PATH) / "runners" / "experiments" / "ScienceBirds" / "SB_experiment"
EXPORT_TRIALS = False   # Export trials xml file
NUM_TRIALS = 1      # Number of trials to run per known/unknown, novelty level and type
PER_TRIAL = 1     # Levels per trial
BEFORE_NOVELTY = 0 # Levels in a trial before novelty is introduced
NOVELTIES = {"1": ["10"]}  # Novelties to use in the experiment (IE, trials to run)>>>>>>> sm/experiments_sb:runners/novelty_experiment_runner_sb.py
#NOVELTIES = {"1": ["6", "7", "8", "9", "10"], "2": ["6", "7", "8", "9", "10"], "3": ["6", "7"]}




def load_lookup(lookup_path):
    with open(lookup_path) as f:
        obj = json.load(f)
        return obj


class NoveltyExperimentRunnerSB:
    """ Runs experiments for science birds using M18 metrics """

    def __init__(self,
                 agent_type: AgentType,
                 novelties: dict = NOVELTIES,
                 num_trials: int = NUM_TRIALS,
                 levels_per_trial: int = PER_TRIAL,
                 levels_before_novelty: int = BEFORE_NOVELTY,
                 export_trials: bool = EXPORT_TRIALS,
                 results_path: pathlib.Path = RESULTS_PATH):
        
        self.agent_type = agent_type
        self.novelties = novelties
        self.num_trials = num_trials
        self.levels_per_trial = levels_per_trial
        self.levels_before_novelty = levels_before_novelty
        self.export_trials = export_trials

        self._results_directory_path = results_path
        if not os.path.exists(self._results_directory_path):
            os.makedirs(self._results_directory_path)

        self.levels = dict()

        # Collect levels
        self.load_levels()

    def load_levels(self):
        """
        Collect levels to be drawn upon when constructing trials
        """

        for novelty_level in NOVELTY_LEVELS:   # Iterate over all novelty levels
            self.levels[novelty_level] = defaultdict(set)

            logger.debug("SB: Loading levels from novelty level {}".format(novelty_level))
            
            novelty_path = os.path.join("Levels", NOVELTY_LEVELS[novelty_level])
            for novelty_type in NOVELTY_TYPES:  # Iterate over all novelty types
                if NOVELTY_TYPES[novelty_type] not in os.listdir(os.path.join(SB_BIN_PATH, novelty_path)):
                    # Skip over types not present
                    continue

                # dir_path = SB_BIN_PATH / novelty_path / NOVELTY_TYPES[novelty_type] / "Levels"
                # print("SB: Loading levels from novelty type {} directory {}".format(novelty_type, dir_path))

                pattern = 'Levels/novelty_level_{}/type{}/Levels/*.xml'.format(novelty_level, novelty_type)
                levels = list(SB_BIN_PATH.glob(pattern))

                for level in levels: # Iterate over all levels of the novelty level and typeå
                    # Add level to set - use shorthand for easier access later
                    self.levels[novelty_level][novelty_type].add(level)

    def reset(self):
        """ reset the experiment """
        self.levels.clear()
        self.load_levels()
    
    def construct_trial(self, novelty_level: str, novelty_type: str, before_novelty: int):
        """
        Construct a trial for a particular novelty level and type.
        Levels will be drawn from the database of levels in SB_BIN_PATH.
        The trial will begin with non-novelty levels, and transition to novelty after the 'before_novelty' index.
        Returns the trial, as well as the novelty id which takes the form '<novelty_level>_<novelty_type>'
        """
        random.seed() # Randomize seed
        
        levels = list()

        for index in range(self.levels_per_trial):
            next_level = ""
            if index < before_novelty: # Collect from non-novel levels
                # TODO: Hardcode level and type for non-novelty for now - we only have non-novel level 0 novelty with type 2
                next_level = self.levels["0"]["2"].pop()
            else:   # Collect from novel levels
                next_level = self.levels[novelty_level][novelty_type].pop()

            levels.append(next_level)

        logger.debug("Created trial: {}".format(["{}\n".format(level) for level in levels]))

        return levels

    def run_trial(self, levels: list, notify: str, trial_num: int, trial_type: str, novelty_level: str, config_file: pathlib.Path = None) -> pandas.DataFrame:
        """ Run a trial """

        notify_novelty = notify == constants.KNOWN

        if config_file is None:
            # Construct the config file
            config = SB_CONFIG_PATH / 'stats_config.xml'

            if self.export_trials:  # Export to unique trial xml config file
                date_time_str = datetime.datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
                config = SB_CONFIG_PATH / "trial_config_{}_{}.xml".format(trial_num, date_time_str)
                logger.debug("Exporting trial to {}".format(config))

            prepare_config(TEMPLATE_PATH, config, levels, notify_novelty)
        else:
            config = config_file

        pre_directories = glob_directories(SB_BIN_PATH, 'Agent*')
        post_directories = None

        # Prepare list to pass by reference - collect agent stats not produced by the SB application
        agent_stats = list()

        # Run the agent
        with run_agent(config, self.agent_type, agent_stats):
            post_directories = glob_directories(SB_BIN_PATH, "Agent*")

        # Identify the results directory that contains all of the .csv result files output by the SB application
        results_directory = diff_directories(pre_directories, post_directories)

        # Collect results
        if results_directory is None:
            post_directories = glob_directories(SB_BIN_PATH, 'Agent*')
            results_directory = diff_directories(pre_directories, post_directories)

        return self.compute_trial_stats(results_directory, agent_stats, trial_num, trial_type,  novelty_level)

    def compute_trial_stats(self,
                            results_directory: str,
                            agent_stats: list,
                            trial_num: int,
                            trial_type: str,
                            novelty_level: str) -> pandas.DataFrame:
        """
        Given a results directory, compute the statistics of the agent's performance.
        agent_stats is a list of dicts - stats of the agent that are not produced by the Angry Birds simulation
        """
        
        trial = pandas.DataFrame(columns=['trial_num', 'trial_type', 'novelty_level', 'episode_type', 'episode_num', 'novelty_probability',
                                          'novelty_threshold', 'novelty', 'novelty_characterization',
                                          'performance', 'pass', 'num_repairs', 'repair_time'])

        # bird_scores = collections.defaultdict(lambda: {"passed": 0, "failed": 0})
        logger.debug("Gathering stats from: {}".format(results_directory))
        # Open results .csv
        evaluation_data = list(results_directory.glob('*_EvaluationData.csv'))
        for eval_data in evaluation_data:
            with open(eval_data) as f:
                data = csv.DictReader(f)
                for episode_num, row in enumerate(data):
                    # birds = get_bird_count(os.path.join(SB_BIN_PATH, level_path))
                    status = row['LevelStatus']
                    
                    is_novelty = constants.NOVELTY

                    if row['novelty'] == 'basic':
                        is_novelty = constants.NON_NOVELTY_PERFORMANCE

                    if 'Pass' not in status:
                        # for bird in birds.keys():
                        #     bird_scores[bird]['passed'] += 1
                        status = 'Fail'
                        # for bird in birds.keys():
                        #     bird_scores[bird]['failed'] += 1

                    score = float(row['Score'])
                    
                    novelty_probability = 0
                    num_repairs = 0
                    repair_time = 0
                    if len(agent_stats) > 0:
                        print(agent_stats)
                        if 'novelty_likelihood' in agent_stats[episode_num]:
                            novelty_probability = agent_stats[episode_num]["novelty_likelihood"]
                        if 'repair_calls' in agent_stats[episode_num]:
                            num_repairs = agent_stats[episode_num]["repair_calls"]
                        if 'repair_time' in agent_stats[episode_num]:
                            repair_time = agent_stats[episode_num]["repair_time"]

                    novelty_characterization = 0    # TODO: find a use for this?
                    novelty_threshold = 1   # TODO: figure out how to extract this
                    novelty = 0     # TODO: This is a value used in Cartpole and not SB domain

                    result = pandas.DataFrame(data={
                        'trial_num': [trial_num],
                        'trial_type': [trial_type],
                        'novelty_level': [novelty_level],
                        'episode_type': [is_novelty],
                        'episode_num': [episode_num],
                        'novelty_probability': [novelty_probability],
                        'novelty_threshold': [novelty_threshold],
                        'novelty': [novelty],
                        'novelty_characterization': [novelty_characterization],
                        'performance': [score],
                        'pass': [status],
                        'num_repairs': [num_repairs],
                        'repair_time': [repair_time]
                    })

                    trial = trial.append(result)

        return trial

    def run_experiment(self, configs: list = None):
        """ Run a set of trials defined by self.novelties OR by a set of config files """
        
        trial_num = 0
        experiment_results = pandas.DataFrame(columns=['trial_num', 'trial_type', 'novelty_level', 'episode_type',
                                                       'episode_num', 'novelty_probability',
                                                       'novelty_threshold', 'novelty', 'novelty_characterization',
                                                       'performance', 'pass', 'num_repairs', 'repair_time'])
        
        experiment_results_path = os.path.join(self._results_directory_path, "novelty_results.csv")
        with open(experiment_results_path, "w+") as f:
            experiment_results.to_csv(f, index=False)

        for _ in range(self.num_trials):
            for trial_type in [constants.UNKNOWN, constants.KNOWN]:
                if configs is not None:
                    for config in configs:
                        logger.debug("Using config file: {}".format)

                        # Get trial details
                        tree = ET.parse(config)

                        was_notified = tree.getroot().find('./trials/trial').get('notify_novelty')
                        novelty_level = "0"

                        level_tags = tree.getroot().find('./trials/trial/game_level_set')
                        for level_tag in level_tags:
                            level_path = level_tag.attrib['level_path']
                            if 'novelty_level_0' in level_path:
                                continue
                            else:
                                novelty_level = level_path.split('/')[1]    # TODO: make this cleaner

                        # Run the trial
                        trial_results = self.run_trial([], was_notified, trial_num, trial_type, novelty_level, config_file=config)
                        experiment_results = experiment_results.append(trial_results)

                        # Export results to file
                        with open(experiment_results_path, "a") as f:
                            trial_results.to_csv(f, index=False, header=False)

                        trial_num += 1
                else:
                    for novelty_level in self.novelties:
                        for novelty_type in self.novelties[novelty_level]:
                            was_notified = constants.NOVELTY

                            if novelty_level in NON_NOVEL_LEVELS:
                                was_notified = constants.NON_NOVELTY_PERFORMANCE

                            levels = self.construct_trial(novelty_level, novelty_type, self.levels_before_novelty)

                            # Run the trial
                            trial_results = self.run_trial(levels, was_notified, trial_num, trial_type, novelty_level)
                            experiment_results = experiment_results.append(trial_results)

                            # Export results to file
                            with open(experiment_results_path, "a") as f:
                                trial_results.to_csv(f, index=False, header=False)

                            trial_num += 1

    @staticmethod
    def categorize_examples_for_novelty_detection(dataframe):
        dataframe['predicted_novel'] = numpy.where(dataframe['novelty_probability'] < dataframe['novelty_threshold'], False, True)
        dataframe['TN'] = numpy.where((dataframe['episode_type'] == constants.NON_NOVELTY_PERFORMANCE) & (dataframe['predicted_novel'] == False), 1, 0)
        dataframe['FP'] = numpy.where((dataframe['episode_type'] == constants.NON_NOVELTY_PERFORMANCE) & (dataframe['predicted_novel'] == True), 1, 0)
        dataframe['TP'] = numpy.where((dataframe['episode_type'] == constants.NOVELTY) & (dataframe['predicted_novel'] == True), 1, 0)
        dataframe['FN'] = numpy.where((dataframe['episode_type'] == constants.NOVELTY) & (dataframe['predicted_novel'] == False), 1, 0)
        return dataframe

    @staticmethod
    def get_trials_summary(dataframe):
        # print(dataframe)
        trials = dataframe[['trial_num', 'trial_type', 'pass', 'FN', 'FP', 'TN', 'TP', 'performance']]
        trials['passed'] = numpy.where(trials['pass'] == 'Pass', 1, 0)
        # print(trials)

        grouped = trials.groupby(['trial_type', 'trial_num'])

        # print("AFTER group output: \n{}".format(grouped))

        aggregate = grouped.agg({'FN': numpy.sum, 'FP': numpy.sum, 'TN': numpy.sum, 'TP': numpy.sum, 'performance': numpy.mean, 'passed': numpy.sum})
        
        # print("AFTER aggregate output: \n{}".format(aggregate))

        aggregate['is_CDT'] = numpy.where((aggregate['TP'] > 1) & (aggregate['FP'] == 0), True, False)
        
        # print("AFTER CDT: {}".format(trials))
        cdt = aggregate[aggregate['is_CDT'] == True]

        return aggregate, cdt

    @staticmethod
    def get_program_metrics(cdt: pandas.DataFrame, trials: pandas.DataFrame, hydra: pandas.DataFrame, baseline: pandas.DataFrame, asymptote: int = 5):
        num_trials_per_type = trials.groupby("trial_type").agg({'FN': len}).rename(columns={'FN': 'count'}) # Get number of trials
        # Aggregate to get
        scores = cdt.groupby("trial_type").agg({'FN': numpy.mean, 'FP': len}).rename(columns={'FN': 'M1', 'FP': 'CDT_count'})

        # M1 - average number of FN among CDTs

        # M2 - % of CDTs
        scores['M2'] = scores['CDT_count'] / num_trials_per_type['count']

        # M2.1 - % of trials with at least 1 FP
        trials['CDT_FP_count'] = numpy.where((trials['FP'] > 0), True, False)

        CDT_FP_count = trials[trials['CDT_FP_count'] == True].groupby("trial_type")
        CDT_FP_count = CDT_FP_count.agg({'CDT_FP_count': len})

        scores['M2.1'] = CDT_FP_count.replace(numpy.nan, 0)
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
                known_avg = group.groupby(["trial_type", "episode_type"]).get_group(("known", "novelty")).agg({'performance': numpy.mean})
                trials_known_avg.append(known_avg)

            if 'unknown' in group.values: # group.trial_type: # M4
                unknown_avg = group.groupby(["trial_type", "episode_type"]).get_group(("unknown", "novelty")).agg({'performance': numpy.mean})
                trials_unknown_avg.append(unknown_avg)

            # M5
            last_per_group = group.groupby("episode_type").get_group("novelty").agg({'performance': numpy.sum})
            trials_last.append(last_per_group)

            # M6
            novelty_grouped = group.groupby("episode_type").get_group("novelty")
            slice = asymptote if len(novelty_grouped.index) > asymptote else len(novelty_grouped.index)
            post_asymptote = novelty_grouped.tail(slice).agg({"performance": numpy.sum})
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
                known_avg = group.groupby(["trial_type", "episode_type"]).get_group(("known", "non-novelty-performance")).agg({'performance': numpy.mean})
                base_known_avg.append(known_avg)

            if 'unknown' in group.values: # group.trial_type:    # M4
                unknown_avg = group.groupby(["trial_type", "episode_type"]).get_group(("unknown", "non-novelty-performance")).agg({'performance': numpy.mean})
                base_unknown_avg.append(unknown_avg)

            # M5
            last_per_group = group.groupby("episode_type").get_group("novelty").agg({'performance': numpy.sum})
            base_last.append(last_per_group)

            # M6
            novelty_grouped = group.groupby("episode_type").get_group("novelty")
            slice = asymptote if len(novelty_grouped.index) > asymptote else len(novelty_grouped.index)
            post_asymptote = novelty_grouped.tail(slice).agg({"performance": numpy.sum})
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
            performance_metrics['M3'] = m3_agg_trials.agg({'known_perf': numpy.mean})
        if not m4_agg_trials.empty and not m4_agg_trials.empty:
            m4_agg_trials['unknown_perf'] = m4_agg_trials[0] / m4_agg_base[0]
            performance_metrics['M4'] = m4_agg_trials.agg({'unknown_perf': numpy.mean})['unknown_perf']
        if not m5_agg_trials.empty and not m5_agg_base.empty:
            m5_agg_trials['opti_t'] = m5_agg_trials[0] / (m5_agg_base[0] + m5_agg_trials[0])
            performance_metrics['M5'] = m5_agg_trials.agg({'opti_t': numpy.sum})[0] / len(m5_agg_trials.index)
        if not m6_agg_trials.empty and not m6_agg_base.empty:
            m6_agg_trials['asymptote'] = m6_agg_trials[0] / m6_agg_base[0].replace({0: numpy.nan})
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
    # experiment_runner.run_experiment(configs=[SB_CONFIG_PATH / "trial_config_1_6.xml"])
