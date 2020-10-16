from agent.consistency.cartpole_repair import CartpoleConsistencyEstimator, CartpoleRepair
from agent.consistency.focused_anomaly_detector import FocusedAnomalyDetector
from agent.consistency.model_formulation import ConsistencyChecker
from agent.consistency.meta_model_repair import *
from agent.planning.cartpole_planner import CartPolePlanner
from agent.planning.cartpole_pddl_meta_model import *
from agent.consistency.observation import CartPoleObservation
from agent.consistency.consistency_estimator import DEFAULT_DELTA_T
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
        self.log = logging.getLogger(__name__).getChild('CartpoleHydraAgent')

        self.observations_list = []
        self.replan_idx = 40

        self.novelty_probability = 0.0
        self.novelty_type = 0
        self.novelty_characterization = dict()

        self.plan_idx = 0
        self.steps = 0
        self.plan = None
        self.current_observation = CartPoleObservation()
        self.last_performance = 0.0

    def episode_end(self, performance: float):
        self.steps = 0
        self.plan_idx = 0
        self.plan = None
        self.observations_list.append(self.current_observation)
        self.current_observation = CartPoleObservation()
        self.last_performance = performance # Records the last performance value, to show impact

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
        self.has_repaired = False
        self.detector = FocusedAnomalyDetector(threshold=[0.012, 0.012, 0.006, 0.009])

    def episode_end(self, performance: float):
        novelties = []
        try:
            novelties = self.detector.detect(self.current_observation)
        except Exception:
            pass

        self.log.info("%d Novelties detected" % len(novelties))
        if (self.has_repaired and performance < self.repair_threshold) or \
                ((not self.has_repaired) and len(novelties) != 0):

            # If this is the first detection of novelty, record the characterization # TODO: Rethink this, it is a hack
            if self.has_repaired==False:
                self.log.info("%d Novelties detected" % len(novelties))

                characterization = dict()
                for focused_anomaly in novelties:
                    # Set the novelty characterization
                    for obs_element in focused_anomaly.obs_elements:
                        if obs_element.property is not None and len(obs_element.property)>0:
                            novelty_properties = obs_element.property.strip().split(" ")
                            for novel_property in novelty_properties:
                                characterization[novel_property] = "Abnormal state attribute"

                self.novelty_probability = 1.0 # TODO:  Replace this with a real prob. estimate
                self.novelty_characterization = characterization

            try:
                meta_model_repair = CartpoleRepair(self.consistency_checker, self.desired_precision)
                repair, _ = meta_model_repair.repair(self.meta_model, self.current_observation, delta_t=DEFAULT_DELTA_T)
                self.log.info("Repaired meta model (repair string: %s)" % repair)
                self.has_repaired = True
            except Exception:
                pass
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
        self.log.debug("Testing instance: sending action={}, novelty_probability={}, novelty_type={}, novelty_characterization={}".format(
            action, novelty_probability, novelty_type, novelty_characterization
        ))
        return action, novelty_probability, novelty_type, novelty_characterization
