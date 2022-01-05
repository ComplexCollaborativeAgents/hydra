import copy
from typing import List

import os
import torch
import numpy as np
from prediction_trainer import Trainer
from sklearn.metrics import mean_squared_error

import pdb
import matplotlib.pyplot as plt
import settings

''' A property of an object in a state'''


class ObsElement:
    def __init__(self, state, object_id, property: str):
        self.state = state
        self.object_id = object_id
        self.property = property


class FocusedAnomaly:
    """ 
    Represents an anomaly. It may consist of several ObsElements because it may be the case that
    each ObsElement by itself is not an anomaly but if they appear together it is. 
    """
    def __init__(self, obs_elements: List[ObsElement]):
        self.obs_elements = obs_elements


class FocusedAnomalyDetector:
    """ A superclass for a focused anomaly detector """
    def __init__(self, threshold=None):
        if threshold is None:
            threshold = [0.004 * 3, 0.002 * 3, 0.0015 * 3, 0.002 * 3]
        self.threshold = threshold  # only anamolies that exceed this threshold are returned

    def detect(self, observation):
        EPSILON = 0.000001  # This is used for cases where the novelty threshold is zero, to avoid divide by zero.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        anomalies = list()
        model = torch.load(os.path.join(settings.ROOT_PATH, 'model', 'model_may2021.pt'), map_location=device)
        prediction_obj = Trainer(model)
        anomaly_count = 10  ## increase this if you want a more conservative detection
        cart_property = np.asarray(["Cart_Position", "Cart_Velocity", "Pole_Angle", "Pole_Angular_Velocity"])
        next_anomaly_idx = 0
        anomaly_list = []
        delta = []

        novelty_likelihood = 0.0
        for i in range(len(observation.states) - 1):
            state = observation.states[i]
            next_state = observation.states[i + 1]
            action = np.array(np.reshape(observation.actions[i], (-1, 1)))
            next_state_pred = prediction_obj.predict(np.array(np.reshape(state, (1, -1))), action)
            current_delta = np.abs(observation.states[i + 1] - next_state_pred.cpu().numpy()[0])
            delta.append(current_delta)
            anomaly_prob = np.mean(np.array(delta), axis=0)

            # Update novelty prob
            normalized_anomaly_scores = [min(1.0, 1.0 - (thresh - anomaly_score + EPSILON) / (thresh + EPSILON))
                                         for (thresh, anomaly_score) in zip(self.threshold, anomaly_prob)]
            anomaly_value = max(normalized_anomaly_scores)
            if anomaly_value > novelty_likelihood:
                novelty_likelihood = anomaly_value

            # If novelty exceeded threshold, add it to the list of detected novelties
            if (anomaly_prob > np.array(self.threshold)).any():
                # if anomaly_value >= 1: TODO: after code freeze, let's be bold and use this one :)

                if next_anomaly_idx < i:  # this checks if the anomalies are contiguous
                    anomaly_list = []
                # computes which state properties are affected by the novelty
                property_idx = np.greater(anomaly_prob, np.array(self.threshold))
                property_type = cart_property[property_idx]
                prop = property_type[0] + " "
                for j in range(len(property_type) - 1):
                    prop += property_type[j + 1] + " "

                anomaly_element = ObsElement(state, None, prop)
                anomaly_list.append(anomaly_element)
                next_anomaly_idx = i + 1

                if len(anomaly_list) == anomaly_count:  ## returns if we have 10/anomaly_count contiguous anomalies
                    print("\n\n\n" + str(anomaly_list))
                    anomalies.append(FocusedAnomaly(
                        copy.deepcopy(anomaly_list)))  # TODO: Replace 1.0 with some funciton of anomaly_prob
                    anomaly_list.clear()
                    break

        # If the trajectory stopped with an anomaly, we also count it as an anomaly # TODO: Discuss this design choice
        if next_anomaly_idx == len(observation.states) - 1 and len(anomaly_list) > 0:
            anomalies.append(FocusedAnomaly(anomaly_list))
        return anomalies, novelty_likelihood
