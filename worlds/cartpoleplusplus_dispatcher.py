import itertools
import logging
import time
from typing import Type, List, Optional

from worlds.wsu.generator.cartpoleplusplus import CartPoleBulletEnv
from worlds.wsu.generator.n_0 import CartPole
# from worlds.wsu.generator.n import CartPoleNoBlocks
from worlds.wsu.wsu_dispatcher import WSUObserver


# Cartpole environment type alias:
CartPoleEnv = Type[CartPoleBulletEnv]


class CartPolePlusPlusDispatcher:
    actions = {'nothing': 0, 'left': 1, 'right': 2, 'forward': 3, 'backward': 4}

    def __init__(self,
                 delegate: WSUObserver,
                 render: bool = False,
                 nominal_env: CartPoleEnv = CartPole,
                 num_episodes: int = 10,
                 pre_novel_episodes: int = 3):
        self.delegate = delegate
        self.render = render
        self.nominal_env = nominal_env
        self.num_episodes = num_episodes
        self.pre_novel_episodes = pre_novel_episodes

        self.log = logging.getLogger(self.__class__.__name__)
        self.log.setLevel(logging.DEBUG)
        self.delegate.set_logger(self.log)
        self.possible_answers = [{'action': a} for a in self.actions.keys()]
        self.delegate.set_possible_answers(self.possible_answers)

    def run(self,
            is_training = False,
            trials: int = 1,
            generators: Optional[List[CartPoleEnv]] = None,
            difficulties: Optional[List[str]] = None,
            informed_trials: bool = True,
            uninformed_trials: bool = True):
        self.delegate.experiment_start()

        if is_training:
            self.delegate.training_start()
            self.__run_training()
            self.delegate.training_end()
            return

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
                self.__run_trial(generator, difficulty, informed=novelty_indicator)
                self.delegate.testing_end()
                self.delegate.trial_end()

        self.delegate.experiment_end()


    def __run_training(self, generator: CartPoleEnv = CartPole, episodes: int = 2000, steps: int = 200):
        self.log.debug("Running training with generator: {}".format(generator.__name__))
        env = generator('easy', renders=False)
        for episode in range(episodes):
            self.delegate.training_episode_start(episode)
            reward = 0
            sum_reward = 0
            done = False
            observation = env.reset()
            time_stamp = time.time()
            features = self.observation_to_feature_vector(observation, env, time_stamp)
            for _ in range(steps):
                if self.render:
                    time.sleep(1/50)

                label = self.delegate.training_instance(feature_vector=features, feature_label=None, reward=reward, done=done)
                #self.log.debug("Received label = {}".format(label))
                action = self.actions[label['action']]

                observation, reward, done, _ = env.step(action)
                reward = 1
                #self.log.debug("Observation: {}, reward: {}, done {}".format(observation, reward, done))
                sum_reward += reward
                time_stamp += 1.0 / 50.0
                features = self.observation_to_feature_vector(observation, env, time_stamp)
                if done:
                    self.delegate.training_instance(feature_vector=features, feature_label=None, reward=reward, done=done)
                    break

            self.delegate.training_episode_end(performance=sum_reward)
            self.log.debug("Episode: {}, total reward {}".format(episode, sum_reward))
        env.close()


    def __run_trial(self,
                    generator: CartPoleEnv,
                    difficulty: str,
                    steps: int = 200,
                    informed: bool = True):
        self.log.debug("Running trial with generator: {} and difficulty: {}".format(generator.__name__, difficulty))
        env = generator(difficulty, renders=self.render)
        for episode in range(self.num_episodes):
            self.delegate.testing_episode_start(episode)
            reward = 0
            time_stamp = time.time()

            if episode == self.pre_novel_episodes and type(env) is not generator:
                env.close()
                env = generator(difficulty, renders=self.render)

            novelty_indicator = not isinstance(env, self.nominal_env) if informed else None

            observation = env.reset()
            features = self.observation_to_feature_vector(observation, env, time_stamp)

            for _ in range(steps):
                if self.render:
                    time.sleep(1/50)

                label = self.delegate.testing_instance(feature_vector=features, novelty_indicator=novelty_indicator)
                # self.log.debug("Received label={}".format(label))
                action = self.actions[label['action']]

                observation, reward, done, _ = env.step(action)

                # WSU is using this time increment:
                # https://github.com/holderlb/WSU-SAILON-NG/blob/master/WSU-Portable-Generator/source/partial_env_generator/test_loader.py#L207
                time_stamp += 1.0 / 50.0

                features = self.observation_to_feature_vector(observation, env, time_stamp)
                if done:
                    break

            self.delegate.testing_episode_end(reward)

        env.close()

    @staticmethod
    def observation_to_feature_vector(observation: dict,
                                      env: CartPoleBulletEnv,
                                      time_stamp: float = 0.0) -> dict:
        features = observation
        features['time_stamp'] = time_stamp
        features['image'] = env.get_image()
        return features
