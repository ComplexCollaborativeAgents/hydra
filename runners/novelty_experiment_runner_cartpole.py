import json

from worlds.gym_cartpole_dispatcher import GymCartpoleDispatcher
from baselines.cartpole.dqn_learner import DQNLearnerObserver
from worlds.wsu.wsu_dispatcher import WSUObserver
from os import path
import settings
import gym
from agent.cartpole_hydra_agent import CartpoleHydraAgent, CartpoleHydraAgentObserver, RepairingCartpoleHydraAgent
import pandas, numpy
import time
from runners import constants
import matplotlib.pyplot as plt
import seaborn as sns


class NoveltyExperimentGymCartpoleDispatcher(GymCartpoleDispatcher):
    def __init__(self, delegate: WSUObserver, model_id: str = 'CartPole-v1', render: bool = False):
        super().__init__(delegate, model_id, render)
        self._env_params = dict()
        self._results = pandas.DataFrame(
            columns=['episode_num', 'novelty_probability', 'novelty_threshold', 'novelty', 'novelty_characterization',
                     'performance'])
        self._is_known = None

    def get_env_params(self):
        return self._env_params

    def set_novelty(self, novelties={}):
        for param in novelties:
            self._env_params[param] = novelties[param]

    def set_is_known(self, is_known=None):
        self._is_known = is_known

    def _make_environment(self):
        environment = gym.make(self.model_id)
        print(environment.env)
        for param in self._env_params:
            setattr(environment.env, param, self._env_params[param])
            assert getattr(environment.env, param) is self._env_params[param]
            print(param, getattr(environment.env, param))
        return environment

    def begin_experiment(self):
        self.delegate.experiment_start()

    def end_experiment(self):
        self.delegate.experiment_end()

    def run_trial(self, trial_number=0, episode_range=range(0, 1)):
        self.delegate.trial_start(trial_number=trial_number, novelty_description={})
        return (self.__run_trial(episodes=episode_range))

    def __run_trial(self, episodes: range = (0, 1), steps: int = 200):
        env = self._make_environment()
        for episode in episodes:
            self.delegate.testing_episode_start(episode)
            rewards = []
            observation = env.reset()
            features = self.observation_to_feature_vector(observation)
            for step in range(1, steps + 1):
                if self.render:
                    env.render()
                    time.sleep(0.05)
                label = self.delegate.testing_instance(feature_vector=features, novelty_indicator=self._is_known)
                print("NOVELTY INDICATOR{}".format(self._is_known))
                self.log.debug("Received label={}".format(label))
                action = self.label_to_action(label)
                observation, reward, done, _ = env.step(action)
                rewards.append(reward)

                features = self.observation_to_feature_vector(observation, 0.02 * step)
                if done:
                    break
            performance = sum(rewards) / float(steps)
            novelty_probability, novelty_threshold, novelty, novelty_characterization = self.delegate.testing_episode_end(
                performance)
            if self._results is not None:
                self._log_data(episode_num=episode, novelty_probability=novelty_probability,
                               novelty_threshold=novelty_threshold, novelty=novelty,
                               novelty_characterization=novelty_characterization, performance=sum(rewards))
        env.close()
        return self._results

    def _log_data(self, episode_num=0, novelty_probability=0, novelty_threshold=0, novelty=None,
                  novelty_characterization=0, performance=0):
        result = pandas.DataFrame(data={
            'episode_num': [episode_num],
            'novelty_probability': [novelty_probability],
            'novelty_threshold': [novelty_threshold],
            'novelty': [novelty],
            'novelty_characterization': [novelty_characterization],
            'performance': [performance / constants.MAX_SCORE]
        }
        )
        self._results = self._results.append(result)


class NoveltyExperimentRunnerCartpole:
    def __init__(self,
                 number_of_experiment_trials=3,
                 non_novelty_learning_trial_length=0,
                 non_novelty_performance_trial_length=3,
                 novelty_trial_length=3):
        self._number_of_experiment_trials = number_of_experiment_trials
        self._non_novelty_learning_trial_length = non_novelty_learning_trial_length
        self._non_novelty_performance_trial_length = non_novelty_performance_trial_length
        self._novelty_trial_length = novelty_trial_length
        self._results_dataframe = pandas.DataFrame(
            columns=['trial_num', 'episode_num', 'trial_type', 'episode_type', 'novelty_probability',
                     'novelty_threshold', 'novelty',
                     'novelty_characterization', 'performance'])

    def run_experiment_subtrial(self, episode_range, trial_num, trial_type, episode_type, novelty_id, novelty):
        observer = CartpoleHydraAgentObserver(agent_type=RepairingCartpoleHydraAgent)
        env_dispatcher = NoveltyExperimentGymCartpoleDispatcher(observer, render=True)

        if trial_type == constants.UNKNOWN:
            env_dispatcher.set_is_known(None)
        else:
            if episode_type == constants.NOVELTY:
                env_dispatcher.set_is_known(True)
            else:
                if episode_type == constants.NON_NOVELTY_PERFORMANCE:
                    env_dispatcher.set_is_known(False)

        if episode_type == constants.NOVELTY:
            env_dispatcher.set_novelty(novelty['config'])

        results = env_dispatcher.run_trial(trial_number=trial_num, episode_range=episode_range)
        results['trial_num'] = trial_num
        results['novelty_id'] = novelty_id
        results['trial_type'] = trial_type
        results['episode_type'] = episode_type
        results['level'] = novelty['level']
        results['env_config'] = str(novelty['config'])
        # return results
        self._results_dataframe = self._results_dataframe.append(results)

    def run_experiment(self, file, novelty_config=None):
        results_file = open(path.join(settings.ROOT_PATH, "data", "cartpole", "test", "repairing_test_wsu.csv"), "a")

        if novelty_config is None:
            novelties_config = NoveltyExperimentRunnerCartpole.generate_novelty_configs()
        else:
            novelties_config = [novelty_config]


        for novelty in novelties_config:
            for trial_type in [constants.UNKNOWN, constants.KNOWN]:
                for trial in range(0, self._number_of_experiment_trials):
                    episode_num = 0
                    self.run_experiment_subtrial(
                        episode_range=range(episode_num, episode_num + self._non_novelty_performance_trial_length),
                        trial_num=trial,
                        trial_type=trial_type,
                        episode_type=constants.NON_NOVELTY_PERFORMANCE,
                        novelty_id=novelty['uid'],
                        novelty=novelty)
                    episode_num = episode_num + self._non_novelty_performance_trial_length
                    self.run_experiment_subtrial(
                        episode_range=range(episode_num, episode_num + self._novelty_trial_length),
                        trial_num=trial,
                        trial_type=trial_type,
                        episode_type=constants.NOVELTY,
                        novelty_id=novelty['uid'],
                        novelty=novelty)
        results_file = open(path.join(settings.ROOT_PATH, "data", "cartpole", "test", "repairing_test_wsu.csv"), "w")
        self._results_dataframe.to_csv(results_file)

    @staticmethod
    def generate_novelty_configs():
        novelty_config = {
            'levels': {
                # 1: ['masscart', 'masspole', 'length'],
                1: [constants.MASSCART],
                2: [constants.GRAVITY]
            },
            'values': range(8, 13)
        }

        novelties = []
        uid = 0
        for level in novelty_config['levels'].keys():
            for param in novelty_config['levels'][level]:
                for value in novelty_config['values']:
                    novelty = {
                        'uid': uid,
                        'level': level,
                        'config': {param: constants.ENV_PARAMS_NOMINAL[param] * value / 10}
                    }
                    novelties.append(novelty)
                    uid = uid + 1
        return novelties

    @staticmethod
    def categorize_examples_for_novelty_detection(dataframe):
        dataframe['is_novel'] = numpy.where(dataframe['novelty_probability'] < dataframe['novelty_threshold'], False,
                                            True)
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
        return scores

    @staticmethod
    def plot_experiment_results(df, novelty_episode_number):
        plt.figure(figsize=(16, 9))
        sns.lineplot(data=df, y='performance', x='episode_num', hue='trial_type', ci=95)
        plt.axvline(x=novelty_episode_number, color='red')
        plt.title("Experiment results", fontsize=20)
        plt.xlabel("episodes", fontsize=15)
        plt.ylabel("performance", fontsize=15)


if __name__ == '__main__':
    experiment_runner = NoveltyExperimentRunnerCartpole(number_of_experiment_trials=5,
                                                        non_novelty_learning_trial_length=0,
                                                        non_novelty_performance_trial_length=5,
                                                        novelty_trial_length=10)
    experiment_runner.run_experiment(novelty_config={'uid': 0, 'level': 1, 'config': {constants.LENGTH: 1}}, file=open(
        path.join(settings.ROOT_PATH, "data", "cartpole", "test", "length_3.csv"), "w"))
