from agent.reward_estimation.architecture import CNN
import torch
import os
import settings
import numpy as np
from agent.reward_estimation.nn_utils.converter import convert_pickle_to_image, normalization, extract_state_action_from_observation
import pickle

REWARD_ESTIMATOR_MODEL_PATH = "{}/model/reward_estimator_model_500_may2022.pth".format(settings.ROOT_PATH)

class RewardEstimator:
    def __init__(self):
        self.model = CNN()
        self.model.load_state_dict(torch.load(REWARD_ESTIMATOR_MODEL_PATH, map_location=torch.device('cpu')) )

    def compute_estimated_reward_difference(self, observation):
        """
        :rtype: difference in estimated expected reward and actual reward upon making the shot
        :param:
        """
        initial_state, applied_action, received_reward = extract_state_action_from_observation(observation)
        initial_state, applied_action, received_reward = normalization(initial_state, applied_action, received_reward)
        state_tensor = torch.tensor(initial_state).float()
        action_tensor = torch.tensor(applied_action).float()
        reward = self.model(state_tensor, action_tensor)[0,:].detach().numpy()
        difference = abs(received_reward[0] - reward[0])
        return difference


if __name__ == "__main__":
    reward_estimator = RewardEstimator()
    obs = pickle.load(open("{}/agent/reward_estimation/test_data/9_3_211011175902_211011_181531_observation.p".format(settings.ROOT_PATH), "rb"))
    print(obs)
    difference = reward_estimator.compute_estimated_reward_difference(obs)
    print(difference)

