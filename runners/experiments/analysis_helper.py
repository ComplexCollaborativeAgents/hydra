import pandas, numpy
import matplotlib.pyplot as plt
import seaborn as sns

class AnalysisHelper:
    @staticmethod
    def categorize_examples_for_novelty_detection(dataframe):
        dataframe['is_novel'] = numpy.where(dataframe['novelty_probability'] < dataframe['novelty_threshold'], False, True)
        dataframe['TN'] = numpy.where((dataframe['episode_type'] == constants.NON_NOVELTY_PERFORMANCE) & (dataframe['is_novel'] == False), 1, 0)
        dataframe['FP'] = numpy.where((dataframe['episode_type'] == constants.NON_NOVELTY_PERFORMANCE) & (dataframe['is_novel'] == True), 1, 0)
        dataframe['TP'] = numpy.where((dataframe['episode_type'] == constants.NOVELTY) & (dataframe['is_novel'] == True),1, 0)
        dataframe['FN'] = numpy.where((dataframe['episode_type'] == constants.NOVELTY) & (dataframe['is_novel'] == False),1, 0)
        return dataframe

    @staticmethod
    def get_trials_summary(dataframe):
        trials = dataframe[['trial_num', 'trial_type', 'novelty_id', 'env_config', 'FN', 'FP', 'TN', 'TP', 'performance']].groupby(
            ['trial_type','novelty_id', 'trial_num']).agg({'FN': numpy.sum, 'FP': numpy.sum, 'TN': numpy.sum, 'TP': numpy.sum, 'performance': numpy.mean})
        trials['is_CDT'] = numpy.where((trials['TP'] > 1) & (trials['FP'] == 0), True, False)
        cdt = trials[trials['is_CDT'] == True]
        return trials, cdt

    @staticmethod
    def get_program_metrics(cdt: pandas.DataFrame, trials: pandas.DataFrame):
        num_trials_per_type = trials.groupby("trial_type").agg({'FN': len}).rename(columns={'FN': 'count'})
        scores = cdt.groupby("trial_type").agg({'FN': numpy.mean, 'FP': len}).rename(columns={'FN': 'M1', 'FP': 'M2'})
        scores['M2'] = scores['M2'] / num_trials_per_type['count']
        scores['NRP'] = trials.groupby("trial_type").agg({'performance': numpy.mean})
        return scores

    @staticmethod
    def plot_experiment_results(df, novelty_episode_number):
        plt.figure(figsize=(16, 9))
        ax = sns.lineplot(data=df, y='performance', x='episode_num', hue='trial_type', ci=95)
        ax.set(ylim=(0, 1.1))
        plt.axvline(x=novelty_episode_number, color='red')
        plt.title("Experiment results", fontsize=20)
        plt.xlabel("episodes", fontsize=15)
        plt.ylabel("performance", fontsize=15)