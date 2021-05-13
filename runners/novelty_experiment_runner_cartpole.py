from worlds.gym_cartpole_dispatcher import GymCartpoleDispatcher
from baselines.cartpole.dqn_learner import DQNLearnerObserver
from worlds.wsu.wsu_dispatcher import WSUObserver
from os import path
import settings
import gym
from agent.cartpole_hydra_agent import CartpoleHydraAgent, CartpoleHydraAgentObserver, RepairingCartpoleHydraAgent
import pandas, numpy
import time
import constants



class NoveltyExperimentGymCartpoleDispatcher(GymCartpoleDispatcher):
    def __init__(self, delegate: WSUObserver, model_id: str = 'CartPole-v1', render: bool = False):
        super().__init__(delegate, model_id, render)
        self._env_params = dict()
        self._results = pandas.DataFrame(
            columns=['episode_num', 'novelty_probability', 'novelty_threshold', 'novelty', 'novelty_characterization',
                     'performance'])
        self._is_known = None

    def set_novelty(self, novelties={}):
        self._env_params = novelties

    def set_is_known(self, is_known: bool = False):
        self._is_known = is_known

    def _make_environment(self):
        environment = gym.make(self.model_id)
        print(environment.env)
        for param in self._env_params:
            value = self._env_params[param]
            setattr(environment.env, param, value)
            assert getattr(environment.env, param) is value
        return environment

    def begin_experiment(self):
        self.delegate.experiment_start()

    def end_experiment(self):
        self.delegate.experiment_end()

    def run_trial(self, episode_range=range(0, 1)):
        self.delegate.trial_start(trial_number=0, novelty_description={})
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
            'novelty_characterization': ['nothing'],
            'performance': [performance]
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
            columns=['trial_num', 'episode_num', 'type', 'novelty_probability', 'novelty_threshold', 'novelty',
                     'novelty_characterization', 'performance'])

    def run_non_novelty_performance_subtrial(self, episode_range, trial_num=0,
                                             novelty={'level': 2, 'config': {'gravity': 10}}):
        observer = CartpoleHydraAgentObserver(agent_type=RepairingCartpoleHydraAgent)
        env_dispatcher = NoveltyExperimentGymCartpoleDispatcher(observer, render=True)
        results = env_dispatcher.run_trial(episode_range=episode_range)
        results['trial_num'] = trial_num
        results['type'] = constants.NON_NOVELTY_PERFORMANCE
        results['level'] = novelty['level']
        param = list(novelty['config'].keys())[0]
        results['param'] = param
        results['value'] = novelty['config'][param]
        self._results_dataframe = self._results_dataframe.append(results)

    def run_unknown_novelty_subtrial(self, episode_range, trial_num=0, novelty={'level': 2, 'config': {'gravity': 10}}):
        observer = CartpoleHydraAgentObserver(agent_type=RepairingCartpoleHydraAgent)
        env_dispatcher = NoveltyExperimentGymCartpoleDispatcher(observer, render=True)
        env_dispatcher.set_novelty(novelty['config'])
        env_dispatcher.set_is_known(False)
        results = env_dispatcher.run_trial(episode_range=episode_range)
        results['trial_num'] = trial_num
        results['type'] = constants.UNKNOWN_NOVELTY
        results['level'] = novelty['level']
        param = list(novelty['config'].keys())[0]
        results['param'] = param
        results['value'] = novelty['config'][param]
        self._results_dataframe = self._results_dataframe.append(results)

    def run_known_novelty_subtrial(self, episode_range, trial_num=0, novelty={'level': 2, 'config': {'gravity': 10}}):
        observer = CartpoleHydraAgentObserver(agent_type=RepairingCartpoleHydraAgent)
        env_dispatcher = NoveltyExperimentGymCartpoleDispatcher(observer, render=True)
        env_dispatcher.set_novelty(novelty['config'])
        env_dispatcher.set_is_known(True)
        results = env_dispatcher.run_trial(episode_range=episode_range)
        results['trial_num'] = trial_num
        results['type'] = constants.KNOWN_NOVELTY
        results['level'] = novelty['level']
        param = list(novelty['config'].keys())[0]
        results['param'] = param
        results['value'] = novelty['config'][param]
        self._results_dataframe = self._results_dataframe.append(results)

    def run_experiment_trial(self, file):
        ## for all novelties in a config file
        for trial in range(0, self._number_of_experiment_trials):
            episode_num = 0
            self.run_non_novelty_performance_subtrial(
                episode_range=range(episode_num, episode_num + self._non_novelty_performance_trial_length),
                trial_num=trial)
            episode_num = episode_num + self._non_novelty_performance_trial_length
            self.run_unknown_novelty_subtrial(
                episode_range=range(episode_num, episode_num + self._novelty_trial_length), trial_num=trial)
            self.run_known_novelty_subtrial(episode_range=range(episode_num, episode_num + self._novelty_trial_length),
                                            trial_num=trial)
        results_file = open(path.join(settings.ROOT_PATH, "data", "cartpole", "test", "repairing_test_wsu.csv"), "w")
        self._results_dataframe.to_csv(results_file)

    @staticmethod
    def generate_novelty_configs():
        level_1_params = ['masscart', 'masspole', 'length']
        level_2_params = ['gravity']
        # level_3_params = ['friction'] not modeled
        values = range(0.5, 2, 0.5)
        print(values)

        pass


if __name__ == '__main__':
    experiment_runner = NoveltyExperimentRunnerCartpole(number_of_experiment_trials=2,
                                                        non_novelty_learning_trial_length=0,
                                                        non_novelty_performance_trial_length=3,
                                                        novelty_trial_length=3)

    experiment_runner.run_experiment_trial(
        file=open(path.join(settings.ROOT_PATH, "data", "cartpole", "test", "repairing_test_wsu.csv"), "w"))

