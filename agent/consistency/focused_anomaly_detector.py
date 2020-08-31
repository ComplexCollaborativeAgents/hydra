from agent.consistency.observation import *
from typing import List,Dict
class ObsElement:
    def __init__(self, state: SBState, obj_id, obj_property):
        self.state = state
        self.obj_id = obj_id
        self.obj_property = obj_property

class FocusedAnomaly:
    def __init__(self, obs_elements : List[ObsElement]):
        self.obs_elements = obs_elements

''' An object that accepts a SC observation and outputs a 
set of FocusedAnomaly objects, each associated with a confidence. '''
class FocusedAnomalyDetector:
    def detect(self, sb_observation: ScienceBirdsObservation):
        anomaly_to_confidence = Dict[FocusedAnomaly,float]

        # Stub implementation
        return anomaly_to_confidence