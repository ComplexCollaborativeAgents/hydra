import logging

import pandas
import os
from worlds.cartpoleplusplus_dispatcher import CartPolePlusPlusDispatcher
from worlds.wsu.wsu_dispatcher import WSUObserver
from worlds.wsu.generator.cartpoleplusplus import CartPoleBulletEnv
from runners import constants



class NoveltyExperimentCartpolePlusPlusDispatcher(CartPolePlusPlusDispatcher):
    def __init__(self,
                 delegate: WSUObserver,
                 render: bool = False,
                 train_with_reward = False,
                 log_details = False,
                 details_directory = None):
        super().__init__(delegate, render)
        self._env_params = dict()
        self._results = pandas.DataFrame(columns=['episode_num', 'novelty_probability', 'novelty_threshold', 'novelty', 'novelty_characterization', 'performance'])
        self._is_known = None
        self._train_with_reward = train_with_reward
        self._log_details = log_details
        self._details_directory = details_directory


    def get_env_params(self):
        return self.__env_params

    def set_novelty(self, novelties={}):
        for param in novelties:
            self._env_params[param] = novelties[param]

    def set_is_known(self, is_known=None):
        self._is_known = is_known

    def _make_environment(self):
        p = self._p
        p.changeDynamics(self.cartpole, 0, mass=20)
        pass

    def begin_experiment(self):
        self.delegate.experiment_start()

    def end_experiment(self):
        self.delegate.experiment_end()

    def __run_trial(self, episodes: range = (0, 1), steps: int = 200):
        env = self._make_environment()
        for episode in episodes:
            self.delegate.testing_episode_start(episode)
            rewards = []
            observation = env.reset()
            features = self.observation_to_feature_vector(observation)
            reward = 0
            done = False
            for step in range(1, steps + 1):
                if self.render:
                    env.render()
                    time.sleep(0.05)
                if self._train_with_reward:
                    label = self.delegate.testing_instance(feature_vector=features, novelty_indicator=self._is_known, reward=reward, done=done)
                else:
                    label = self.delegate.testing_instance(feature_vector=features, novelty_indicator=self._is_known)
                self.log.debug("Received label={}".format(label))
                action = self.label_to_action(label)
                observation, reward, done, _ = env.step(action)
                rewards.append(reward)

                features = self.observation_to_feature_vector(observation, 0.02 * step)
                if done:
                    break
            performance = sum(rewards) / float(steps)
            if self._log_details:
                self.log_details(episode)
            novelty_probability, novelty_threshold, novelty, novelty_characterization = self.delegate.testing_episode_end(performance)


            if self._results is not None:
                self._log_data(episode_num=episode, novelty_probability=novelty_probability,
                               novelty_threshold=novelty_threshold, novelty=novelty,
                               novelty_characterization=novelty_characterization, performance=sum(rewards))
        env.close()
        return self._results

    def log_details(self, episode):
        if not os.path.exists(self._details_directory):
            os.makedirs(self._details_directory)
        file = open(os.path.join(self._details_directory, "episode_{}.csv".format(str(episode))), 'w')
        log_df = pandas.DataFrame(data={
            'states': self.delegate.agent.states,
            'actions': self.delegate.agent.actions,
            'cnn_likelihood': self.delegate.agent.cnn_likelihoods
        })
        log_df.to_csv(file)
        file.close()

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
