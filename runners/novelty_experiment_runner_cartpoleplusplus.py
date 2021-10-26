import logging
import time
import itertools
import settings

logging.basicConfig(format='%(name)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("hydra_agent")
import pandas
from typing import Type, List, Optional

import os
from worlds.cartpoleplusplus_dispatcher import CartPolePlusPlusDispatcher
from worlds.wsu.wsu_dispatcher import WSUObserver
from worlds.wsu.generator.cartpoleplusplus import CartPoleBulletEnv
from runners import constants
from agent.cartpole_hydra_agent import CartpoleHydraAgent, CartpoleHydraAgentObserver, RepairingCartpoleHydraAgent
from worlds.wsu.generator.m_1 import CartPolePPMock1
from worlds.wsu.generator.n_0 import CartPole
from worlds.wsu.generator.cartpoleplusplus import CartPoleBulletEnv
CartPoleEnv = Type[CartPoleBulletEnv]


class NoveltyExperimentRunnerCartpolePlusPlusDispatcher(CartPolePlusPlusDispatcher):
    def __init__(self,
                 delegate: WSUObserver,
                 render: bool = False,
                 nominal_env: CartPoleEnv = CartPole,
                 num_episodes: int = 10,
                 pre_novel_episodes: int = 3):
        super().__init__(delegate, render, nominal_env, num_episodes, pre_novel_episodes)
        self._results_directory_path = os.path.join(settings.ROOT_PATH, "runners", "experiments", "cartpole_plusplus","test")
        if not os.path.exists(self._results_directory_path):
            os.makedirs(self._results_directory_path)

        self._results_file_path = os.path.join(self._results_directory_path, "basic.csv")
        self.log.debug("Writing at {}".format(self._results_file_path))
        self._results_file_handle = open(self._results_file_path, 'w')
        self._results = pandas.DataFrame(columns=['trial_num',
                                                  'novelty_id',
                                                  'trial_type',
                                                  'episode_type',
                                                  'level',
                                                  'env_config',
                                                  'episode_num',
                                                  'novelty_probability',
                                                  'novelty_threshold',
                                                  'novelty',
                                                  'novelty_characterization',
                                                  'performance'])
        self._results.to_csv(self._results_file_handle, index=False)

    def run(self,
            trials: int = 1,
            generators: Optional[List[CartPoleEnv]] = None,
            difficulties: Optional[List[str]] = None,
            informed_trials: bool = True,
            uninformed_trials: bool = True):
        self.delegate.experiment_start()
        self.delegate.training_start()
        # generate training data here in the future, if needed
        self.delegate.training_end()

        self.log.debug("generators {}".format(generators))

        if generators is None:
            generators = [self.nominal_env]
        if difficulties is None:
            difficulties = ['easy']

        novelty_indicators = []
        if informed_trials:
            novelty_indicators.append(True)
        if uninformed_trials:
            novelty_indicators.append(False)

        for novelty_indicator, generator, difficulty in itertools.product(novelty_indicators, generators, difficulties):
            if novelty_indicator:
                self.log.debug("Running informed novelty trials")
            else:
                self.log.debug("Running uninformed novelty trials")

            for trial in range(trials):
                self.delegate.trial_start(trial, dict())
                self.delegate.testing_start()
                results_df = self.__run_trial(generator, difficulty, informed=novelty_indicator)
                results_df['trial_num'] = trial
                results_df['novelty_id'] = 0
                results_df['trial_type'] = novelty_indicator
                results_df['level'] = 0
                results_df['env_config'] = 'none'
                self.delegate.testing_end()
                self.delegate.trial_end()
                self._results = self._results.append(results_df)

        self._results.to_csv(self._results_file_handle, index=False, header=False)
        self.delegate.experiment_end()

    def __run_trial(self,
                    generator: CartPoleEnv,
                    difficulty: str,
                    steps: int = 200,
                    informed: bool = True):
        self.log.debug("Running trial with generator: {} and difficulty: {}".format(generator.__name__, difficulty))
        results_df = pandas.DataFrame(columns=['episode_num',
                                               'novelty_probability',
                                               'novelty_threshold',
                                               'novelty',
                                               'novelty_characterization',
                                               'performance'])
        env = self.nominal_env(difficulty, renders=self.render)
        for episode in range(self.num_episodes):
            self.delegate.testing_episode_start(episode)
            rewards = []
            time_stamp = time.time()
            episode_type = 'non-novelty-performance'

            if episode == self.pre_novel_episodes and type(env) is not generator:
                env.close()
                env = generator(difficulty, renders=self.render)
                episode_type = 'novelty'

            novelty_indicator = not isinstance(env, self.nominal_env) if informed else None

            observation = env.reset()
            features = self.observation_to_feature_vector(observation, env, time_stamp)

            for _ in range(steps):
                if self.render:
                    time.sleep(1 / 50)

                label = self.delegate.testing_instance(feature_vector=features, novelty_indicator=novelty_indicator)
                self.log.debug("Received label={}".format(label))
                action = self.actions[label['action']]

                observation, reward, done, _ = env.step(action)
                rewards.append(reward)

                # WSU is using this time increment:
                # https://github.com/holderlb/WSU-SAILON-NG/blob/master/WSU-Portable-Generator/source/partial_env_generator/test_loader.py#L207
                time_stamp += 1.0 / 30.0

                features = self.observation_to_feature_vector(observation, env, time_stamp)
                if done:
                    break

            novelty_probability, novelty_threshold, novelty, novelty_characterization = self.delegate.testing_episode_end(reward)
            result = pandas.DataFrame(data={
                'episode_num': [episode],
                'novelty_probability': [novelty_probability],
                'novelty_threshold': [novelty_threshold],
                'novelty': [novelty],
                'episode_type': [episode_type],
                'novelty_characterization': [novelty_characterization],
                'performance': [sum(rewards) / constants.MAX_SCORE]
            })
            results_df = results_df.append(result)
        env.close()
        return results_df


class NoveltyExperimentRunnerCartpolePlusPlus:
    def __init__(self):
        self._pre_novel_episodes = 2
        self._number_episodes = 2
        self._num_trials = 1
        self._generators = [CartPolePPMock1]
        self._difficulties=['easy']
        self._observer = WSUObserver()

    def run(self):
        env = NoveltyExperimentRunnerCartpolePlusPlusDispatcher(self._observer,
                                         render=True,
                                         num_episodes=self._number_episodes,
                                         pre_novel_episodes=self._pre_novel_episodes)
        env.run(generators=self._generators,
                difficulties=self._difficulties,
                trials=self._num_trials)


if __name__ == '__main__':
    runner = NoveltyExperimentRunnerCartpolePlusPlus()
    runner.run()


