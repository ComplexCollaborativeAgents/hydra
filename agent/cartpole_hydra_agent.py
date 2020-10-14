from agent.consistency.cartpole_repair import CartpoleConsistencyEstimator, CartpoleRepair
from agent.consistency.model_formulation import ConsistencyChecker
from agent.consistency.meta_model_repair import *
from agent.planning.cartpole_planner import CartPolePlanner
from agent.planning.cartpole_pddl_meta_model import *
from agent.consistency.observation import CartPoleObservation
import time
import copy
import numpy as np
import settings
import random
from typing import Type

from worlds.wsu.wsu_dispatcher import WSUObserver


class CartpoleHydraAgent:
    def __init__(self):
        self.consistency_checker = ConsistencyChecker()
        self.meta_model = CartPoleMetaModel()
        self.cartpole_planner = CartPolePlanner(self.meta_model)

        self.observations_list = []
        self.replan_idx = 40

        self.novelty_probability = 0.0
        self.novelty_type = 0
        self.novelty_characterization = dict()

        self.plan_idx = 0
        self.steps = 0
        self.plan = None
        self.current_observation = CartPoleObservation()

    def episode_end(self, performance: float):
        self.steps = 0
        self.plan_idx = 0
        self.plan = None
        self.observations_list.append(self.current_observation)
        self.current_observation = CartPoleObservation()

    def testing_instance(self, feature_vector: dict, novelty_indicator: bool = None) -> \
            (dict, float, int, dict):

        observation = self.feature_vector_to_observation(feature_vector)
        if self.plan is None:
            self.meta_model.constant_numeric_fluents['time_limit'] = 4.0
            self.plan = self.cartpole_planner.make_plan(observation, 0)

        if self.plan_idx >= self.replan_idx:
            self.meta_model.constant_numeric_fluents['time_limit'] = round((4.0 - ((self.steps - 1) * 0.02)), 2)
            new_plan = self.cartpole_planner.make_plan(observation, 0)
            if len(new_plan) != 0:
                self.plan = new_plan
                self.plan_idx = 0

        action = random.randint(0, 1)
        if self.plan_idx < len(self.plan):
            action = 1 if self.plan[self.plan_idx].action_name == "move_cart_right dummy_obj" else 0

        self.plan_idx += 1
        self.steps += 1
        self.current_observation.actions.append(action)
        self.current_observation.states.append(observation)

        label = self.action_to_label(action)
        return label, self.novelty_probability, self.novelty_type, self.novelty_characterization

    @staticmethod
    def feature_vector_to_observation(feature_vector: dict) -> np.ndarray:
        # sometimes the is a typo in the feature vector we get from WSU:
        if 'cart_veloctiy' in feature_vector: feature_vector['cart_velocity'] = feature_vector['cart_veloctiy']

        features = ['cart_position', 'cart_velocity', 'pole_angle', 'pole_angular_velocity']
        return np.array([feature_vector[f] for f in features])

    @staticmethod
    def action_to_label(action: int) -> dict:
        labels = [{'action': 'left'}, {'action': 'right'}]
        return labels[action]


class RepairingCartpoleHydraAgent(CartpoleHydraAgent):
    def __init__(self):
        super().__init__()
        self.consistency_checker = CartpoleConsistencyEstimator()
        self.desired_precision = 0.01
        self.repair_threshold = 0.975 # 195/200

    def episode_end(self, performance: float):
        if performance < self.repair_threshold:
            meta_model_repair = CartpoleRepair(self.consistency_checker, self.desired_precision)
            repair, _ = meta_model_repair.repair(self.meta_model, self.current_observation, delta_t=DEFAULT_DELTA_T)
        super().episode_end(performance)


class CartpoleHydraAgentObserver(WSUObserver):
    def __init__(self, agent_type: Type[CartpoleHydraAgent] = CartpoleHydraAgent):
        super().__init__()
        self.agent_type = agent_type
        self.agent = None

    def trial_start(self, trial_number: int, novelty_description: dict):
        super().trial_start(trial_number, novelty_description)
        self.agent = self.agent_type()

    def testing_episode_end(self, performance: float):
        super().testing_episode_end(performance)
        self.agent.episode_end(performance)

    def testing_instance(self, feature_vector: dict, novelty_indicator: bool = None) -> \
            (dict, float, int, dict):
        super().testing_instance(feature_vector, novelty_indicator)
        action, novelty_probability, novelty_type, novelty_characterization = \
            self.agent.testing_instance(feature_vector, novelty_indicator)
        return action, novelty_probability, novelty_type, novelty_characterization
