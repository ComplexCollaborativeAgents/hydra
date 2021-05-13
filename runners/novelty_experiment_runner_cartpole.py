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



class NoveltyExperimentGymCartpoleDispatcher(GymCartpoleDispatcher):
    def __init__(self, delegate: WSUObserver, model_id: str = 'CartPole-v1', render: bool = False):
        super().__init__(delegate, model_id, render)
        self._env_params = {
            'gravity': 9.8,
            'masscart': 1,
            'masspole': 0.1,
            'length': 0.5,
            'force_mag': 10.0
        }
        self._results = pandas.DataFrame(
            columns=['episode_num', 'novelty_probability', 'novelty_threshold', 'novelty', 'novelty_characterization',
                     'performance'])
        self._is_known = None

    def get_env_params(self):
        return self._env_params

    def set_novelty(self, novelties={}):
        for param in novelties:
            value = self._env_params[param]
            self._env_params[param] = value * novelties[param]

    def set_is_known(self, is_known: bool = False):
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

    # def run_non_novelty_performance_subtrial(self, episode_range, trial_num,
    #                                          novelty={'level': 2, 'config': {'gravity': 0.9}}):
    #     observer = CartpoleHydraAgentObserver(agent_type=RepairingCartpoleHydraAgent)
    #     env_dispatcher = NoveltyExperimentGymCartpoleDispatcher(observer, render=True)
    #     results = env_dispatcher.run_trial(episode_range=episode_range)
    #     results['trial_num'] = trial_num
    #     results['type'] = constants.NON_NOVELTY_PERFORMANCE
    #     results['level'] = novelty['level']
    #     results['env_config'] = str(env_dispatcher.get_env_params())
    #     self._results_dataframe = self._results_dataframe.append(results)
    #
    # def run_unknown_novelty_subtrial(self, episode_range, trial_num, novelty={'level': 2, 'config': {'gravity': 0.9}}):
    #     observer = CartpoleHydraAgentObserver(agent_type=RepairingCartpoleHydraAgent)
    #     env_dispatcher = NoveltyExperimentGymCartpoleDispatcher(observer, render=True)
    #     env_dispatcher.set_novelty(novelty['config'])
    #     env_dispatcher.set_is_known(False)
    #     results = env_dispatcher.run_trial(episode_range=episode_range)
    #     results['trial_num'] = trial_num
    #     results['type'] = constants.UNKNOWN_NOVELTY
    #     results['level'] = novelty['level']
    #     results['env_config'] = str(env_dispatcher.get_env_params())
    #     self._results_dataframe = self._results_dataframe.append(results)
    #
    # def run_known_novelty_subtrial(self, episode_range, trial_num, novelty={'level': 2, 'config': {'gravity': 0.9}}):
    #     observer = CartpoleHydraAgentObserver(agent_type=RepairingCartpoleHydraAgent)
    #     env_dispatcher = NoveltyExperimentGymCartpoleDispatcher(observer, render=True)
    #     env_dispatcher.set_novelty(novelty['config'])
    #     env_dispatcher.set_is_known(True)
    #     results = env_dispatcher.run_trial(episode_range=episode_range)
    #     results['trial_num'] = trial_num
    #     results['type'] = constants.KNOWN_NOVELTY
    #     results['level'] = novelty['level']
    #     results['env_config'] = str(env_dispatcher.get_env_params())
    #     self._results_dataframe = self._results_dataframe.append(results)

    def run_experiment_subtrial(self, episode_range, trial_num, type, novelty_id, novelty):
        observer = CartpoleHydraAgentObserver(agent_type=RepairingCartpoleHydraAgent)
        env_dispatcher = NoveltyExperimentGymCartpoleDispatcher(observer, render=True)
        if type == constants.KNOWN_NOVELTY or type == constants.UNKNOWN_NOVELTY:
            env_dispatcher.set_novelty(novelty['config'])
        if type == constants.KNOWN_NOVELTY:
            env_dispatcher.set_is_known(True)
        else:
            env_dispatcher.set_is_known(False)
        results = env_dispatcher.run_trial(episode_range=episode_range)
        results['trial_num'] = trial_num
        results['novelty_id'] = novelty_id
        results['type'] = type
        results['level'] = novelty['level']
        results['env_config'] = str(env_dispatcher.get_env_params())
        #return results
        self._results_dataframe = self._results_dataframe.append(results)


    def run_experiment(self, file):
        results_file = open(path.join(settings.ROOT_PATH, "data", "cartpole", "test", "repairing_test_wsu.csv"), "a")
        novelties = NoveltyExperimentRunnerCartpole.generate_novelty_configs()
        novelty_id = 0
        for novelty in novelties:
            for trial in range(0, self._number_of_experiment_trials):
                episode_num = 0
                self.run_experiment_subtrial(
                    episode_range=range(episode_num, episode_num + self._non_novelty_performance_trial_length),
                    trial_num=trial,
                    type=constants.NON_NOVELTY_PERFORMANCE,
                    novelty_id=novelty_id,
                    novelty=novelty)
                #results.to_csv(results_file)
                episode_num = episode_num + self._non_novelty_performance_trial_length
                self.run_experiment_subtrial(
                    episode_range=range(episode_num, episode_num + self._non_novelty_performance_trial_length),
                    trial_num=trial,
                    type=constants.UNKNOWN_NOVELTY,
                    novelty_id=novelty_id,
                    novelty=novelty)
                #results.to_csv(results_file)
                self.run_experiment_subtrial(
                    episode_range=range(episode_num, episode_num + self._non_novelty_performance_trial_length),
                    trial_num=trial,
                    type=constants.KNOWN_NOVELTY,
                    novelty_id=novelty_id,
                    novelty=novelty)
                #results.to_csv(results_file)
            novelty_id += 1
        results_file = open(path.join(settings.ROOT_PATH, "data", "cartpole", "test", "repairing_test_wsu.csv"), "w")
        self._results_dataframe.to_csv(results_file)

    @staticmethod
    def generate_novelty_configs():
        novelty_config = {
            'levels': {
                #1: ['masscart', 'masspole', 'length'],
                1: ['masscart'],
                2: ['gravity']
            },
            'values': range(8, 13)
        }

        novelties = []
        for level in novelty_config['levels'].keys():
            for param in novelty_config['levels'][level]:
                for value in novelty_config['values']:
                    novelty = {
                        'level': level,
                        'config': {param: value / 10}
                    }
                    novelties.append(novelty)
        return novelties


if __name__ == '__main__':
    experiment_runner = NoveltyExperimentRunnerCartpole(number_of_experiment_trials=1,
                                                        non_novelty_learning_trial_length=0,
                                                        non_novelty_performance_trial_length=1,
                                                        novelty_trial_length=1)
    experiment_runner.run_experiment(file=open(path.join(settings.ROOT_PATH, "data", "cartpole", "test", "repairing_test_wsu.csv"), "w"))
