import logging
import time
import random

import numpy
import gym

from worlds.wsu.wsu_dispatcher import WSUObserver


class GymCartpoleDispatcher:

    def __init__(self, delegate: WSUObserver, model_id: str = 'CartPole-v1', render: bool = False):
        self.delegate = delegate
        self.model_id = model_id
        self.render = render

        self.log = logging.getLogger(__name__).getChild('GymCartpole')
        self.log.setLevel(logging.DEBUG)
        self.delegate.set_logger(self.log)
        self.possible_answers = [{'action': 'left'}, {'action': 'right'}]
        self.delegate.set_possible_answers(self.possible_answers)

    def run(self, trials: int = 1):
        self.delegate.experiment_start()
        self.delegate.training_start()
        # generate training data here in the future
        self.delegate.training_end()

        for trial in range(trials):
            self.delegate.trial_start(trial, dict())
            self.delegate.testing_start()
            self.__run_trial()
            self.delegate.testing_end()
            self.delegate.trial_end()

        self.delegate.experiment_end()

    def __run_trial(self, episodes: int = 5, steps: int = 200):
        env = self._make_environment()
        for episode in range(episodes):
            self.delegate.testing_episode_start(episode)
            rewards = []
            observation = env.reset()
            features = self.observation_to_feature_vector(observation)

            for step in range(1, steps + 1):
                if self.render:
                    env.render()
                    time.sleep(0.05)

                label = self.delegate.testing_instance(feature_vector=features, novelty_indicator=None)
                self.log.debug("Received label={}".format(label))
                action = self.label_to_action(label)
                observation, reward, done, _ = env.step(action)
                rewards.append(reward)

                features = self.observation_to_feature_vector(observation, 0.02 * step)
                if done:
                    break

            performance = sum(rewards) / float(steps)
            self.delegate.testing_episode_end(performance)
        env.close()

    def _make_environment(self):
        env = gym.make(self.model_id)
        # env.env.force_mag = 8
        return env

    @staticmethod
    def observation_to_feature_vector(observation: numpy.ndarray, time: float = 0.0) -> dict:
        return {'time_stamp': time,
                'cart_position': observation[0],
                'cart_velocity': observation[1],
                'pole_angle': observation[2],
                'pole_angular_velocity': observation[3]}

    @staticmethod
    def label_to_action(label: dict) -> int:
        if 'left' in label['action']:
            return 0
        else:
            return 1
