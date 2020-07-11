import pickle
import subprocess

import settings
from agent.perception.perception import Perception
from agent.planning.pddl_meta_model import MetaModel
from agent.planning.pddl_plus import PddlPlusPlan
from worlds.science_birds import SBState

''' A small object that represents an observation of the SB game, containing the values
(state, action, intermedidate_states, reward)'''
class ScienceBirdsObservation:
    def __init__(self):
        self.state = None # An SBState
        self.action = None  # an SBAction
        self.intermediate_states = None # The  sequence of intermediates states observed after doing the action
        self.reward = 0 # The reward obtained from performing an action


    ''' Returns a sequence of PDDL states that are the observed intermediate states '''
    def get_trace(self, meta_model: MetaModel = MetaModel()): # TODO: Refactor and move this to the meta model?
        observed_state_seq = []
        perception = Perception()
        for intermediate_state in self.intermediate_states:
            if isinstance(intermediate_state, SBState):
                intermediate_state = perception.process_sb_state(intermediate_state)
            observed_state_seq.append(meta_model.create_pddl_state(intermediate_state))
        return observed_state_seq

    ''' Returns a PDDL+ plan object with a single action that is the action that was performed '''
    def get_pddl_plan(self, meta_model: MetaModel = MetaModel):
        pddl_plan = PddlPlusPlan()
        pddl_plan.append(meta_model.create_timed_action(self.action, self.state))
        return pddl_plan

    ''' Stores the observation in a file specified by the prefix'''
    def log_observation(self,prefix):
        trace_dir = "{}/agent/consistency/trace/observations".format(settings.ROOT_PATH)
        cmd = "mkdir -p {}".format(trace_dir)
        subprocess.run(cmd, shell=True)
        pickle.dump(self, open("{}/{}_observation.p".format(trace_dir,prefix), 'wb'))

    def load_observation(full_path):
        return pickle.load(open(full_path, 'rb'))