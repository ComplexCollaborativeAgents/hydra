import logging
import math
import random
import time

import numpy as np

from agent.consistency.observation import CartPolePlusPlusObservation
from agent.planning.cartpoleplusplus_pddl_meta_model import CartPolePlusPlusMetaModel
from agent.planning.cartpoleplusplus_planner import CartPolePlusPlusPlanner
from agent.repair.cartpoleplusplus_repair import CartpolePlusPlusRepair, CartpolePlusPlusConsistencyEstimator
from agent.consistency.focused_anomaly_detector import FocusedAnomalyDetector
import json
import settings
from typing import Type

from worlds.wsu.wsu_dispatcher import WSUObserver
from agent.hydra_agent import HydraAgent

class CartpolePlusPlusHydraAgent(HydraAgent):
    def __init__(self):
        super().__init__(planner=CartPolePlusPlusPlanner(CartPolePlusPlusMetaModel()), meta_model_repair=None)

        self.log = logging.getLogger(__name__).getChild('CartpolePlusPlusHydraAgent')

        self.observations_list = []
        self.default_replan_idx = 5
        self.replan_idx = self.default_replan_idx

        self.novelty_likelihood = 0.0
        self.novelty_existence = None

        self.novelty_type = 0
        self.novelty_characterization = {}
        self.novelty_threshold = 0.999999

        self.recorded_novelty_likelihoods = []
        self.consistency_scores = []

        self.plan_idx = 0
        self.steps = 0
        self.plan = None
        self.current_observation = CartPolePlusPlusObservation()
        self.last_performance = []
        self.episode_timer = time.time()

    def episode_end(self, performance: float, feedback: dict = None):
        self.steps = 0
        self.plan_idx = 0
        self.plan = None
        self.episode_timer = time.time()
        self.observations_list.append(self.current_observation)
        self.current_observation = CartPolePlusPlusObservation()
        self.last_performance.append(performance) # Records the last performance value, to show impact

        return self.novelty_likelihood, self.novelty_threshold, self.novelty_type, self.novelty_characterization

    def choose_action(self, observation: CartPolePlusPlusObservation) -> \
            dict:

        euls = self.quaternions_to_eulers(observation['pole']['x_quaternion'], observation['pole']['y_quaternion'],
                                          observation['pole']['z_quaternion'], observation['pole']['w_quaternion'])

        if round(abs(math.degrees(euls[0])), 6) > 3.0 or round(abs(math.degrees(euls[1])), 6) > 3.0:
            self.replan_idx = 5
        elif round(abs(math.degrees(euls[0])), 6) > 5.0 or round(abs(math.degrees(euls[1])), 6) > 5.0:
            self.replan_idx = 2
        else:
            self.replan_idx = self.default_replan_idx

        if self.plan is None:
            # self.meta_model.constant_numeric_fluents['time_limit'] = 4.0
            self.meta_model.constant_numeric_fluents['time_limit'] = max(0.02, min(4.0, round((4.0 - ((self.steps) * 0.02)), 2)))
            self.plan = self.planner.make_plan(observation, 0)
            self.current_observation = CartPolePlusPlusObservation()
            if len(self.plan) == 0:
                self.plan_idx = 999

        if (self.plan_idx >= self.replan_idx) and ((time.time() - self.episode_timer) < settings.CP_EPISODE_TIME_LIMIT):
            self.meta_model.constant_numeric_fluents['time_limit'] = max(0.02, min(4.0, round((4.0 - ((self.steps) * 0.02)), 2)))
            new_plan = self.planner.make_plan(observation, 0)
            if len(new_plan) != 0:
                self.current_observation = CartPolePlusPlusObservation()
                self.plan = new_plan
                self.plan_idx = 0

        # state_values_list = self.planner.extract_state_values_from_trace("%s/plan_cartpole_prob.pddl" % str(settings.CARTPOLEPLUSPLUS_PLANNING_DOCKER_PATH))
        # state_values_list.insert(0, (observation['cart']['x_position'], observation['cart']['y_position'], observation['cart']['x_velocity'], observation['cart']['y_velocity'],
        #                              euls[0], euls[1], observation['pole']['x_velocity'], observation['pole']['y_velocity']))
        # if (len(state_values_list) > 1):
        #     print("cart observation (X,Y,Vx,Vy):\t\t" + str(observation['cart']['x_position']) + ",\t\t " + str(observation['cart']['y_position']) +
        #           ",\t\t " + str(observation['cart']['x_velocity']) + ",\t\t " + str(observation['cart']['y_velocity']))
        #     print("cart plan val (X,Y,Vx,Vy):\t\t\t" + str(state_values_list[self.plan_idx][0]) + ",\t\t " + str(state_values_list[self.plan_idx][1]) +
        #           ",\t\t " + str(state_values_list[self.plan_idx][2]) + ",\t\t " + str(state_values_list[self.plan_idx][3]))
        #
        #     print("pole observation (X,Y,Vx,Vy):\t" + str(round(math.degrees(euls[0]), 6)) + ",\t\t " + str(round(math.degrees(euls[1]), 6)) +
        #           ",\t\t " + str(observation['pole']['x_velocity']) + ",\t\t " + str(observation['pole']['y_velocity']))
        #     print("pole plan val (X,Y,Vx,Vy):\t\t" + str(
        #         round(math.degrees(state_values_list[self.plan_idx][4]), 6)) + ",\t\t " + str(
        #         round(math.degrees(state_values_list[self.plan_idx][5]), 6)) + ",\t\t " + str(
        #         state_values_list[self.plan_idx][6]) + ",\t\t " + str(state_values_list[self.plan_idx][7]))
        # print("STEP: " + str(self.steps))

        action = random.randint(0, 4)
        if self.plan_idx < len(self.plan):
            action = 0
            if self.plan[self.plan_idx].action_name == "do_nothing dummy_obj":
                action = 0
            elif self.plan[self.plan_idx].action_name == "move_cart_left dummy_obj":
                action = 1
            elif self.plan[self.plan_idx].action_name == "move_cart_right dummy_obj":
                action = 2
            elif self.plan[self.plan_idx].action_name == "move_cart_forward dummy_obj":
                action = 3
            elif self.plan[self.plan_idx].action_name == "move_cart_backward dummy_obj":
                action = 4

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

        features = ['cart', 'pole', 'blocks', 'time_stamp', 'image']
        #
        # if 'walls' in feature_vector:
        #     features = ['cart', 'pole', 'blocks', 'walls', 'time_stamp', 'image']
        return np.array([feature_vector[f] for f in features])

    @staticmethod
    def action_to_label(action: int) -> dict:
        labels = [{'action': 'nothing'}, {'action': 'left'}, {'action': 'right'}, {'action': 'forward'}, {'action': 'backward'}]
        return labels[action]

    @staticmethod
    def quaternions_to_eulers(x, y, z, w):
        ysqr = y * y

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + ysqr)
        X = np.arctan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = np.where(t2 > +1.0, +1.0, t2)
        # t2 = +1.0 if t2 > +1.0 else t2

        t2 = np.where(t2 < -1.0, -1.0, t2)
        # t2 = -1.0 if t2 < -1.0 else t2
        Y = np.arcsin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (ysqr + z * z)
        Z = np.arctan2(t3, t4)

        return (X, Y, Z)

class RepairingCartpolePlusPlusHydraAgent(CartpolePlusPlusHydraAgent):
    def __init__(self):
        super().__init__()
        self.repair_threshold = 0.975 # 195/200
        self.has_repaired = False
        self.consistency_checker = CartpolePlusPlusConsistencyEstimator()
        self.meta_model_repair = CartpolePlusPlusRepair(self.meta_model, consistency_checker=self.consistency_checker)

    def episode_end(self, performance: float, feedback: dict = None)-> \
            (float, float, int, dict):
        super().episode_end(performance) # Update


        novelty_likelihood, novelty_characterization, has_repaired = self.novelty_detection()
        self.novelty_likelihood = novelty_likelihood
        self.novelty_characterization = novelty_characterization
        self.has_repaired = has_repaired

        self.recorded_novelty_likelihoods.append(self.novelty_likelihood)

        return self.novelty_likelihood, self.novelty_threshold, self.novelty_type, self.novelty_characterization


    def should_repair(self, observation: CartPolePlusPlusObservation) -> bool:
        ''' Checks if we should repair basd on the given observation '''
        return self.novelty_existence is not False and (len(self.last_performance) >= 2) and (self.last_performance[-1] < self.repair_threshold) and (self.last_performance[-2] < self.repair_threshold)

    def novelty_detection(self):
        ''' Computes the likelihood that the current observation is novel '''
        novelty_likelihood = self.novelty_likelihood
        novelty_characterization = self.novelty_characterization
        has_repaired = False

        last_observation = self.observations_list[-1]

        if self.should_repair(last_observation):
            novelty_characterization, novelty_likelihood = self.repair_meta_model(last_observation)
            has_repaired = True


        if self.novelty_existence is True:
            novelty_likelihood = 1.0

        return novelty_likelihood, novelty_characterization, has_repaired

    def repair_meta_model(self, last_observation):
        ''' Repair the meta model based on the last observation '''

        novelty_likelihood = self.novelty_likelihood
        novelty_characterization = self.novelty_characterization

        try:
            repair, consistency = self.meta_model_repair.repair(last_observation,
                                                           delta_t=settings.CP_DELTA_T)
            self.log.info("Repaired meta model (repair string: %s)" % repair)
            nonzero = any(map(lambda x: x != 0, repair))
            if nonzero:
                novelty_likelihood = 1.0
                self.has_repaired = True
                novelty_characterization = json.dumps(dict(zip(self.meta_model_repair.fluents_to_repair, repair)))
            elif consistency > settings.CP_CONSISTENCY_THRESHOLD:
                novelty_likelihood = 1.0
                novelty_characterization = json.dumps({'Unknown novelty': 'no adjustments made'})
            self.consistency_scores.append(consistency)
        except Exception:
            pass
        return novelty_characterization, novelty_likelihood


class CartpolePlusPlusHydraAgentObserver(WSUObserver):
    def __init__(self, agent_type: Type[CartpolePlusPlusHydraAgent] = CartpolePlusPlusHydraAgent):
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

        self.agent.novelty_existence = novelty_indicator

        # observation = self.agent.feature_vector_to_observation(feature_vector)
        observation = feature_vector

        action = self.agent.choose_action(observation)

        # self.log.debug("Testing instance: sending action={}".format(action))
        return action
