from typing import List


''' A property of an object in a state'''
class ObsElement:
    def __init__(self, state, object_id, property:str):
        self.state = state
        self.object_id = object_id
        self.property = property

''' Represents an anomaly. It may consist of several ObsElements because it may be the case that 
each ObsElement by itself is not an anomaly but if they appear together it is. '''
class FocusedAnomaly:
    def __init__(self, obs_elements: List[ObsElement]):
        self.obs_elements = obs_elements

''' A superclass for a focused anomaly detector '''
class FocusedAnomalyDetector():
    def __init__(self,threshold):
        self.threshold = threshold # only anamolies that exceed this threshold are returned

    def detect(self, observation):
        anomaly_to_confidence = dict()
        # Here will be a code that fills the dictionary
        return anomaly_to_confidence


