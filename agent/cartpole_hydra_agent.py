import logging
import random

import numpy as np

from agent.consistency.observation import CartPoleObservation
from agent.planning.cartpole_meta_model import CartPoleMetaModel
from agent.planning.cartpole_planner import CartPolePlanner
from agent.repair.cartpole_repair import CartpoleRepair
from agent.consistency.focused_anomaly_detector import FocusedAnomalyDetector
import json
import settings
from typing import Type

from worlds.wsu.wsu_dispatcher import WSUObserver
from agent.hydra_agent import HydraAgent

# Dictionary mapping AI Gym environment parameter names to PDDL+ fluent names
ENV_PARAM_TO_FLUENT = {
    'gravity':'gravity',
    'force_mag':'force_mag',
    'length':'l_pole',
    'masscart':'m_cart',
    'masspole': 'm_pole',
    'x_threshold':'x_limit',
    'theta_threshold_radians':'angle_limit'
}

class CartpoleHydraAgent(HydraAgent):
    def __init__(self):
        super().__init__(planner=CartPolePlanner(CartPoleMetaModel()), meta_model_repair=None)

        self.log = logging.getLogger(__name__).getChild('CartpoleHydraAgent')
        self.log.setLevel(logging.DEBUG)

        self.observations_list = []
        self.replan_idx = 25

        self.novelty_likelihood = 0.0
        self.novelty_existence = None
        self.novelty_info=None # Allowing the dispatcher to add additional info about the novelty (to distinguish M3/4)

        self.novelty_type = 0
        self.novelty_characterization = {}
        self.novelty_threshold = 0.999999

        self.recorded_novelty_likelihoods = []
        self.consistency_scores = []

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
        return self.novelty_likelihood, self.novelty_threshold, self.novelty_type, self.novelty_characterization

    def choose_action(self, state) -> \
            dict:

        if self.plan is None:
            # self.meta_model.constant_numeric_fluents['time_limit'] = 4.0
            self.meta_model.constant_numeric_fluents['time_limit'] = max(0.02, min(4.0, round((4.0 - ((self.steps) * 0.02)), 2)))
            self.log.info(f"Planning (first plan)...")
            print(f"Planning (first plan)...")
            self.plan = self.planner.make_plan(state, 0)
            self.current_observation = CartPoleObservation()
            if len(self.plan) == 0:
                self.plan_idx = 999

        if self.plan_idx >= self.replan_idx:
            self.meta_model.constant_numeric_fluents['time_limit'] = max(0.02, min(4.0, round((4.0 - ((self.steps) * 0.02)), 2)))
            self.log.info(f"Replanning ...")
            print(f"Planning (plan index={self.plan_idx})...")
            new_plan = self.planner.make_plan(state, 0)
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
        self.current_observation.states.append(state)

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
        self.meta_model_repair = CartpoleRepair()

    def episode_end(self, performance: float, feedback: dict = None)-> \
            (float, float, int, dict):
        super().episode_end(performance) # Update

        novelty_likelihood, novelty_characterization = self.novelty_detection()
        self.novelty_likelihood = novelty_likelihood
        self.novelty_characterization = novelty_characterization

        self.recorded_novelty_likelihoods.append(self.novelty_likelihood)

        return self.novelty_likelihood, self.novelty_threshold, self.novelty_type, self.novelty_characterization


    def should_repair(self, observation: CartPoleObservation) -> bool:
        ''' Checks if we should repair basd on the given observation '''
        return self.novelty_existence is not False and self.last_performance[-1] < self.repair_threshold


    def novelty_detection(self):
        ''' Computes the likelihood that the current observation is novel '''
        last_observation = self.observations_list[-1]
        novelty_likelihood = 0
        novelty_characterization = {}

        if self.should_repair(last_observation):
            self.log.info("Initiating repair...")
            novelty_characterization, novelty_likelihood = self.repair_meta_model(last_observation)

        if self.novelty_existence is True:
            novelty_likelihood = 1.0

        return novelty_likelihood, novelty_characterization

    def _update_meta_model_with_novelty_info(self):
        ''' Update self.meta_model with the given novelty info '''
        new_repairable_constants = []
        new_repair_delta = []
        for repair_env_param, repair_value in self.novelty_info.items():
            if repair_env_param not in ENV_PARAM_TO_FLUENT:
                self.log.info(f"Repair env parameter {repair_env_param} is not mapped to any fluent in the PDDL+ model")
                continue  # Repairable constant is not a known fluent in our PDDL+
            repair_fluent_name = ENV_PARAM_TO_FLUENT[repair_env_param]
            for i, repair_constant in enumerate(self.meta_model.repairable_constants):
                if repair_fluent_name == repair_constant:
                    # Set the constant value of the repairable meta model, if we have it
                    if repair_value is not None:
                        original_value = self.meta_model.constant_numeric_fluents[repair_fluent_name]
                        repair_delta = abs(original_value - repair_value)  # Delta must be positive
                    else:
                        repair_delta = self.meta_model.repair_deltas[i]

                    if abs(repair_delta)>0.0001:
                        # Only attempt a repair if the delta is non-zero
                        new_repairable_constants.append(repair_fluent_name)
                        new_repair_delta.append(repair_delta)
                        self.log.info(f"Set repair parameter {repair_fluent_name} with delta {repair_delta}")
                    else:
                        self.log.info(f"Repair delta for parameter {repair_fluent_name} was zero - do not attempt to repair it")
        # Set the repair constants and deltas in the meta model
        self.meta_model.repairable_constants = new_repairable_constants
        self.meta_model.repair_deltas = new_repair_delta

    def repair_meta_model(self, last_observation):
        ''' Repair the meta model based on the last observation '''
        novelty_characterization = dict()
        try:
            # If novelty info exists, this provides hints for what the repair should be
            if self.novelty_info is not None:
                self.log.info(f"Identified novelty info {self.novelty_info} - adapt meta-model repair accordingly")
                self._update_meta_model_with_novelty_info()

            # Set the repair constants and deltas in the meta_model_repair object
            repair, consistency = self.meta_model_repair.repair(self.meta_model, last_observation,
                                                           delta_t=settings.CP_DELTA_T)
            self.log.info("Repaired meta model (repair string: %s)" % repair)
            nonzero = any(map(lambda x: x != 0, repair))
            if nonzero:
                novelty_likelihood = 1.0
                self.has_repaired = True
                novelty_characterization = self.meta_model_repair.get_repair_as_json(repair)
            elif consistency > settings.CP_CONSISTENCY_THRESHOLD:
                novelty_likelihood = 1.0
                novelty_characterization = json.dumps({'Unknown novelty': 'no adjustments made'})
            else:
                novelty_likelihood = consistency / settings.CP_CONSISTENCY_THRESHOLD
                novelty_characterization = {}
            self.consistency_scores.append(consistency)
        except Exception as err:
            self.log.exception(err)
            pass
        return novelty_characterization, novelty_likelihood


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

    def testing_instance(self, feature_vector: dict, novelty_indicator: bool = None, novelty_info=None) -> \
            dict:
        super().testing_instance(feature_vector, novelty_indicator)

        self.agent.novelty_existence = novelty_indicator
        if "novelty_info" in dir(self.agent):
            self.agent.novelty_info = novelty_info

        state = self.agent.feature_vector_to_observation(feature_vector)  # TODO: Rename this function to feature_vector_to_state to avoid confusion with CartpoleObservation

        action = self.agent.choose_action(state)
        self.log.info("Testing instance: sending action={}".format(action))
        return action
