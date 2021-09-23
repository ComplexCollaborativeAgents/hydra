import logging
logging.basicConfig(format='%(name)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("hydra_agent")
import settings
from machin.frame.algorithms import DQN
import torch
import torch.nn as nn
from worlds.wsu.wsu_dispatcher import WSUObserver
from typing import Type
import os
import numpy as np
from scipy.spatial.transform import Rotation

from worlds.cartpoleplusplus_dispatcher import CartPolePlusPlusDispatcher

BASELINE_CARTPOLE_MODEL_PATH = os.path.join(settings.ROOT_PATH, "baselines", "cartpole", "dqn_learner_cartpoleplusplus.py")

class QNet(nn.Module):
    def __init__(self, observation_space_dim, action_space_dim):
        super(QNet, self).__init__()

        self.fc1 = nn.Linear(observation_space_dim, 55)
        self.fc2 = nn.Linear(55, 55)
        #self.fc3
        self.fc4 = nn.Linear(55, action_space_dim)

    def forward(self, state):
        a = torch.relu(self.fc1(state))
        a = torch.relu(self.fc2(a))
        return self.fc3(a)


class DQNLearner():
    def __init__(self,
                 action_space=5,
                 observation_space=11,
                 model_weights=None,
                 load_from_path=False):
        self._action_space = action_space
        self._observation_space = observation_space
        self._previous_state = None
        self._previous_action = None
        self.initialize_qnet(load_from_path)

    def initialize_qnet(self, load_from_path):
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self._device = 'cpu'
        #print("Running the dqn learner on {}".format(self._device))

        if load_from_path:
            print("Attempting to load DQN model from {}".format(BASELINE_CARTPOLE_MODEL_PATH))
            self._dqn = torch.load(BASELINE_CARTPOLE_MODEL_PATH)
        else:
            self._dqn = QNet(self._observation_space, self._action_space).to(self._device)
        self._dqn_handler = DQN(self._dqn, self._dqn, torch.optim.Adam,
                                nn.MSELoss(reduction='sum'), mode="vanilla")
        self.reset()

    def generate_state(self, observation):
        state_tensor = torch.tensor(observation, dtype=torch.float32).view(1, self._observation_space).to(self._device)
        return state_tensor

    def reset(self):
        self._previous_state = None
        self._previous_action = None
        pass

    def set_epsilon_zero(self):
        pass

    def get_next_action(self, observation, reward, done, is_training):
        state = self.generate_state(observation)
        if is_training:
            action = self._dqn_handler.act_discrete_with_noise({"state": state})
            if self._previous_state is not None:
                self._dqn_handler.store_transition({"state": {"state": self._previous_state},
                                                    "action": {"action": self._previous_action},
                                                    "next_state": {"state": state},
                                                    "reward": reward, "terminal": done})

                self._dqn_handler.update()
                #print("updated dqn")
            self._previous_state = state
            self._previous_action = action
        else:
            action = self._dqn_handler.act_discrete({"state": state})
        return action.item()


    def save_model(self):
        torch.save(self._dqn, BASELINE_CARTPOLE_MODEL_PATH)
        print("saved model at {}".format(BASELINE_CARTPOLE_MODEL_PATH))

    def load_model(self):
        self._dqn = torch.load_state_dict(torch.load(BASELINE_CARTPOLE_MODEL_PATH))

class DQNAgentRunner():
    def __init__(self, environment, agent):
        self._environment = environment
        self._agent = agent

    def run(self, number_of_episodes=200, is_training=True, steps_per_episode=1000):
        scores = []
        for e in range(number_of_episodes):
            observation = self._environment.reset()
            self._agent.reset()
            reward = None
            done = False
            score = 0
            steps = 0
            while True:
                #action = self._agent.get_next_action(observation, reward, done, is_training=is_training)
                action = self._agent.get_next_action(observation, 0, False, is_training=False)
                if done or steps >= steps_per_episode:
                    break
                observation, reward, done, filler = self._environment.step(action)
                score += reward
                steps += 1
            print("training in episode {}, score {}".format(e, score))
            scores.append(score)
        print(scores)



class DQNLearnerObserver(WSUObserver):
    def __init__(self, agent_type: Type[DQNLearner] = DQNLearner, load_from_path=True):
        super().__init__()
        self.log = logging.getLogger(__name__).getChild('DQNLearner')
        self.agent_type = agent_type
        self.agent = None
        self.load_from_path = load_from_path
        self.training_episode = 0

    def trial_start(self, trial_number: int, novelty_description: dict):
        super().trial_start(trial_number, novelty_description)
        self.agent = self.agent_type(load_from_path=False)
        self.agent.reset()

    def training_start(self):
        self.agent = self.agent_type(load_from_path=False)
        self.log.debug("Starting training the DQN network")

    def training_end(self):
        self.log.debug("Training completed")

    def training_episode_start(self, episode_number: int):
        self.training_episode = episode_number
        self.agent.reset()

    def training_episode_end(self, performance: float, feedback: dict = None) -> \
            (float, float, int, dict):
        self.log.debug("Score earned this training episode {}: {}".format(self.training_episode, performance))

    def training_instance(self, feature_vector: dict, feature_label: dict, reward: float, done: bool) ->  \
            dict:
        observation = DQNLearnerObserver.feature_vector_to_observation(feature_vector)
        action_vector =  self.agent.get_next_action(observation, reward, done, is_training=True)
        return DQNLearnerObserver.action_to_label(action_vector)

    def testing_instance(self, feature_vector: dict, novelty_indicator: bool = None, reward=None, done=None) -> \
            dict:
        super().testing_instance(feature_vector, novelty_indicator)
        observation = DQNLearnerObserver.feature_vector_to_observation(feature_vector)
        #print("novelty_indicator {} reward {} done {}".format(novelty_indicator, reward, done))
        try:
            if novelty_indicator and reward:
                action_vector = self.agent.get_next_action(observation, reward, done, is_training=True)
            else:
                action_vector = self.agent.get_next_action(observation, reward, done, is_training=False)
        except:
            action_vector = self.agent.get_next_action(observation, reward, done, is_training=False)

        action = DQNLearnerObserver.action_to_label(action_vector)
        self.log.debug("Testing instance: sending action={}".format(action))
        return action

    @staticmethod
    def action_to_label(action: int) -> dict:
        labels = [{'action': 'nothing'}, {'action': 'left'}, {'action': 'right'}, {'action': 'forward'}, {'action': 'backward'}]
        return labels[action]

    @staticmethod
    def feature_vector_to_observation(feature_vector: dict) -> np.ndarray:
        array = np.array([feature_vector['cart']['x_position'],
                          feature_vector['cart']['y_position'],
                          feature_vector['cart']['x_velocity'],
                          feature_vector['cart']['y_velocity'],
                          feature_vector['pole']['x_quaternion'],
                          feature_vector['pole']['y_quaternion'],
                          feature_vector['pole']['z_quaternion'],
                          feature_vector['pole']['w_quaternion'],
                          feature_vector['pole']['x_velocity'],
                          feature_vector['pole']['y_velocity'],
                          feature_vector['pole']['z_velocity']])
        return array





if __name__ == '__main__':
    observer = DQNLearnerObserver(agent_type=DQNLearner, load_from_path=False)
    env = CartPolePlusPlusDispatcher(observer, render=True)
    env.run(is_training=True)

    # import gym
    # env = gym.make('CartPole-v1')
    # #agent = DQNLearner(action_space = env.action_space, observation_space=env.observation_space)
    # agent = DQNLearner(action_space=env.action_space.n, observation_space=env.observation_space.shape[0], load_from_path=True)
    # runner = DQNAgentRunner(environment=env, agent=agent)
    # #runner.run(is_training=True)
    # runner.run(is_training=False, steps_per_episode=200)
    # #agent.save_model()


