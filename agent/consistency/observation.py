import pickle
import subprocess

import settings
from agent.perception.perception import Perception
from agent.planning.pddl_meta_model import MetaModel
from agent.planning.cartpole_pddl_meta_model import CartPoleMetaModel
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

    def get_initial_state(self):
        return self.state

    ''' Returns a sequence of PDDL states that are the observed intermediate states '''
    def get_trace(self, meta_model: MetaModel = MetaModel()): # TODO: Refactor and move this to the meta model?
        observed_state_seq = []
        perception = Perception()
        for intermediate_state in self.intermediate_states:
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



''' A small object that represents an observation of the SB game, containing the values
(state, action, intermedidate_states, reward)'''
class CartPoleObservation:
    def __init__(self):
        self.states = [] # An SBState
        self.actions = []  # an SBAction
        self.rewards = [] # The reward obtained from performing an action

    def get_initial_state(self):
        return self.states[0]

    ''' Returns a sequence of PDDL states that are the observed intermediate states '''
    def get_trace(self, meta_model: CartPoleMetaModel = CartPoleMetaModel()): # TODO: Refactor and move this to the meta model?
        observed_state_seq = []
        for state in self.states:
            pddl = meta_model.create_pddl_state(state)
            observed_state_seq.append(pddl)
        return observed_state_seq

    ''' Returns a PDDL+ plan object with a single action that is the action that was performed '''
    def get_pddl_plan(self, meta_model: CartPoleMetaModel = CartPoleMetaModel):
        pddl_plan = PddlPlusPlan()
        previous_action_name = "move_cart_right dummy_obj" # TODO: Better to get the default side from the meta model, but also better to discuss design
        for ix in range(len(self.actions)):
            timed_action = meta_model.create_timed_action(self.actions[ix], ix)
            if timed_action.action_name!=previous_action_name:
                pddl_plan.append(timed_action)
                previous_action_name = timed_action.action_name
            # print("\n\nOBSERVATION ACTIONS: ")
            # print(ix, " - ", self.actions[ix])
        return pddl_plan

    ''' Stores the observation in a file specified by the prefix'''
    def log_observation(self,prefix):
        trace_dir = "{}/agent/consistency/trace/observations".format(settings.ROOT_PATH)
        cmd = "mkdir -p {}".format(trace_dir)
        subprocess.run(cmd, shell=True)
        pickle.dump(self, open("{}/{}_observation.p".format(trace_dir,prefix), 'wb'))

    def load_observation(full_path):
        return pickle.load(open(full_path, 'rb'))