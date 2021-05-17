from agent.repair.cartpole_repair import CartpoleRepair
from agent.consistency.focused_anomaly_detector import FocusedAnomalyDetector
from agent.planning.cartpole_planner import CartPolePlanner
from agent.planning.cartpole_pddl_meta_model import *
from agent.consistency.observation import CartPoleObservation
import json
import numpy as np
import settings
import random
from typing import Type
import time

from worlds.wsu.wsu_dispatcher import WSUObserver


class CartpoleHydraAgent:
    def __init__(self):
        self.meta_model = CartPoleMetaModel()
        self.cartpole_planner = CartPolePlanner(self.meta_model)
        self.log = logging.getLogger(__name__).getChild('CartpoleHydraAgent')

        self.observations_list = []
        self.replan_idx = 25

        self.novelty_probability = 0.0
        self.novelty_type = 0
        self.novelty_characterization = {}
        self.novelty_threshold = 0.999999

        self.recorded_novelty_likelihoods = []
        self.consistency_scores = []
        self.novelty_existence = None

        self.plan_idx = 0
        self.steps = 0
        self.plan = None
        self.current_observation = CartPoleObservation()
        self.last_performance = []

    def episode_end(self, performance: float, feedback: dict = None):
        self.steps = 0
        self.plan_idx = 0
        self.plan = None
        self.observations_list.append(self.current_observation)
        self.current_observation = CartPoleObservation()
        self.last_performance.append(performance) # Records the last performance value, to show impact
        return self.novelty_probability, self.novelty_threshold, self.novelty_type, self.novelty_characterization

    def testing_instance(self, feature_vector: dict, novelty_indicator: bool = None) -> \
            dict:

        self.novelty_existence = novelty_indicator

        observation = self.feature_vector_to_observation(feature_vector)



        if self.plan is None:
            # self.meta_model.constant_numeric_fluents['time_limit'] = 4.0
            self.meta_model.constant_numeric_fluents['time_limit'] = max(0.02, min(4.0, round((4.0 - ((self.steps) * 0.02)), 2)))
            self.plan = self.cartpole_planner.make_plan(observation, 0)
            self.current_observation = CartPoleObservation()
            if len(self.plan) == 0:
                self.plan_idx = 999

        if self.plan_idx >= self.replan_idx:
            self.meta_model.constant_numeric_fluents['time_limit'] = max(0.02, min(4.0, round((4.0 - ((self.steps) * 0.02)), 2)))
            new_plan = self.cartpole_planner.make_plan(observation, 0)
            self.current_observation = CartPoleObservation()
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
        return label

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
        self.repair_threshold = 0.975 # 195/200
        self.has_repaired = False
        self.detector = FocusedAnomalyDetector()

    def episode_end(self, performance: float, feedback: dict = None)-> \
            (float, float, int, dict):
        # novelties = []
        # novelty_likelihood=0.0
        #
        # try:
        #     novelties, novelty_likelihood = self.detector.detect(self.current_observation)
        #     self.recorded_novelty_likelihoods.append(novelty_likelihood)
        # except Exception:
        #     pass
        #
        # self.log.info("%d Novelties detected" % len(novelties))
        # if (performance < self.repair_threshold) and (self.novelty_existence != False) and ((self.novelty_existence == True) or (self.has_repaired) or \
        #         ((not self.has_repaired) and len(self.recorded_novelty_likelihoods)>4 and (sum(self.recorded_novelty_likelihoods[-5:])/5 > 0.99))):
        #
        #     # If this is the first detection of novelty, record the characterization # TODO: Rethink this, it is a hack
        #     if self.has_repaired==False:
        #         self.log.info("%d Novelties detected" % len(novelties))
        #
        #         characterization = dict()
        #         for focused_anomaly in novelties:
        #             # Set the novelty characterization
        #             for obs_element in focused_anomaly.obs_elements:
        #                 if obs_element.property is not None and len(obs_element.property)>0:
        #                     novelty_properties = obs_element.property.strip().split(" ")
        #                     for novel_property in novelty_properties:
        #                         characterization[novel_property] = "Abnormal state attribute"
        #
        #         # if novelty_likelihood==0.0:
        #         #     novelty_likelihood = 1.0 # Novelty likelihood is zero when novelty detector says so, but we set this to novelty for some other reasons
        #         self.novelty_characterization['novelty_characterization_description'] = json.dumps(characterization)
        #
        #     try:
        #         meta_model_repair = CartpoleRepair()
        #         repair, _ = meta_model_repair.repair(self.meta_model, self.current_observation, delta_t=settings.CP_DELTA_T)
        #         self.log.info("Repaired meta model (repair string: %s)" % repair)
        #         self.has_repaired = True
        #     except Exception:
        #         pass
        #
        # self.novelty_probability = max(self.novelty_probability, novelty_likelihood)

        if self.novelty_existence is not False and performance < self.repair_threshold:
            try:
                meta_model_repair = CartpoleRepair()
                repair, consistency = meta_model_repair.repair(self.meta_model, self.current_observation, delta_t=settings.CP_DELTA_T)
                self.log.info("Repaired meta model (repair string: %s)" % repair)
                nonzero = any(map(lambda x: x != 0, repair))
                if nonzero:
                    self.novelty_probability = 1.0
                    self.has_repaired = True
                    self.novelty_characterization = json.dumps(dict(zip(meta_model_repair.fluents_to_repair,repair)))
                elif consistency > settings.CP_CONSISTENCY_THRESHOLD:
                    self.novelty_characterization = json.dumps({'Unknown novelty':'no adjustments made'})
                    self.novelty_probability = 1.0
                self.consistency_scores.append(consistency)
            except Exception:
                pass

        if self.novelty_existence is True:
            self.novelty_probability = 1.0

        self.recorded_novelty_likelihoods.append(self.novelty_probability)
        return super().episode_end(performance)


class CartpoleHydraAgentObserver(WSUObserver):
    def __init__(self, agent_type: Type[CartpoleHydraAgent] = CartpoleHydraAgent):
        super().__init__()
        self.agent_type = agent_type
        self.agent = None

    def trial_start(self, trial_number: int, novelty_description: dict):
        super().trial_start(trial_number, novelty_description)
        self.agent = self.agent_type()

    def testing_episode_end(self, performance: float, feedback: dict = None) -> \
            (float, float, int, dict):
        super().testing_episode_end(performance, feedback)
        return self.agent.episode_end(performance, feedback)

    def testing_instance(self, feature_vector: dict, novelty_indicator: bool = None) -> \
            dict:
        super().testing_instance(feature_vector, novelty_indicator)
        action = self.agent.testing_instance(feature_vector, novelty_indicator)
        self.log.debug("Testing instance: sending action={}".format(action))
        return action
