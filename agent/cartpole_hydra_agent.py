from agent.consistency.model_formulation import ConsistencyChecker
from agent.planning.cartpole_planner import CartPolePlanner
from agent.planning.cartpole_pddl_meta_model import *
from agent.consistency.observation import CartPoleObservation
import time
import copy
import numpy as np
import settings
import random

from worlds.wsu.wsu_dispatcher import WSUObserver


class CartpoleHydraAgent:

    def __init__(self):
        self.consistency_checker = ConsistencyChecker()
        self.meta_model = CartPoleMetaModel()
        self.cartpole_planner = CartPolePlanner(self.meta_model)
        self.novelty_likelihood = 0.0
        self.observations_list = []
        self.debug_info = False
        self.plan_idx = 0
        self.plan = None
        self.replan_idx = 20

    def testing_instance(self, feature_vector: dict, novelty_indicator: bool = None) -> \
            (dict, bool, int):

        observation = self.feature_vector_to_observation(feature_vector)
        if self.plan is None:
            self.plan = self.cartpole_planner.make_plan(observation, 0)

        if self.plan_idx >= self.replan_idx:
            new_plan = self.cartpole_planner.make_plan(observation, 0)
            if len(new_plan) != 0:
                self.plan = new_plan
                self.plan_idx = 0

        action = random.randint(0,1)
        if self.plan_idx < len(self.plan):
            action = 1 if self.plan[self.plan_idx].action_name == "move_cart_right dummy_obj" else 0

        # todo: store observations
        self.plan_idx += 1
        label = self.action_to_label(action)
        novelty = False
        novelty_type = 0
        return label, novelty, novelty_type

    @staticmethod
    def feature_vector_to_observation(feature_vector: dict) -> np.ndarray:
        features = ['cart_position', 'cart_velocity', 'pole_angle', 'pole_angular_velocity']
        return np.array([feature_vector[f] for f in features])

    @staticmethod
    def action_to_label(action: int) -> dict:
        labels = [{'action': 'left'}, {'action': 'right'}]
        return labels[action]


class CartpoleHydraAgentObserver(WSUObserver):
    def __init__(self):
        super().__init__()
        self.agent = None

    def testing_episode_start(self, episode_number: int):
        super().testing_episode_start(episode_number)
        self.agent = CartpoleHydraAgent()

    def testing_instance(self, feature_vector: dict, novelty_indicator: bool = None) -> \
            (dict, bool, int):
        super().testing_instance(feature_vector, novelty_indicator)
        action, novelty, novelty_type = self.agent.testing_instance(feature_vector, novelty_indicator)
        return action, novelty, novelty_type
