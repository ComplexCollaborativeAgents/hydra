import pickle
import subprocess

import settings
from agent.planning.sb_meta_model import ScienceBirdsMetaModel
from agent.planning.cartpole_meta_model import CartPoleMetaModel
from agent.planning.pddl_plus import PddlPlusPlan
from agent.planning.meta_model import MetaModel

class HydraObservation:
    ''' An object representsing an observation of a full episode. This includes a trajectory of states and actions and a reward '''

    def get_initial_state(self):
        ''' Returns the first state in this observation '''
        raise NotImplementedError()

    def get_pddl_states_in_trace(self, meta_model: MetaModel):
        ''' Returns a sequence of PDDL states that are the observed intermediate states '''
        raise NotImplementedError()

    def get_pddl_plan(self, meta_model: MetaModel) -> PddlPlusPlan:
        ''' Returns a PDDL+ plan object with a single action that is the action that was performed '''
        raise NotImplementedError()

    def log_observation(self,prefix):
        ''' Stores the observation in a file specified by the prefix'''
        trace_dir = "{}/agent/consistency/trace/observations".format(settings.ROOT_PATH)
        cmd = "mkdir -p {}".format(trace_dir)
        subprocess.run(cmd, shell=True)
        pickle.dump(self, open("{}/{}_observation.p".format(trace_dir,prefix), 'wb'))


class ScienceBirdsObservation(HydraObservation):
    ''' An object that represents an observation of the SB game '''

    def __init__(self):
        self.state = None # An SBState
        self.action = None  # an SBAction
        self.intermediate_states = None # The  sequence of intermediates states observed after doing the action
        self.reward = 0  # The reward obtained from performing an action

    def get_initial_state(self):
        return self.state

    def get_pddl_states_in_trace(self, meta_model: ScienceBirdsMetaModel = ScienceBirdsMetaModel()) -> list: # TODO: Refactor and move this to the meta model?
        ''' Returns a sequence of PDDL states that are the observed intermediate states '''
        observed_state_seq = []
        for intermediate_state in self.intermediate_states:
            observed_state_seq.append(meta_model.create_pddl_state(intermediate_state))
        return observed_state_seq

    def get_pddl_plan(self, meta_model: ScienceBirdsMetaModel = ScienceBirdsMetaModel):
        ''' Returns a PDDL+ plan object with a single action that is the action that was performed '''
        pddl_plan = PddlPlusPlan()
        pddl_plan.append(meta_model.create_timed_action(self.action, self.state))
        return pddl_plan

    def hasUnknownObj(self):
        if self.state.novel_objects():
            return True
        else:
            return False

    def get_novel_object_ids(self):
        ''' Return a list of novel object ids '''
        return [object_id for [object_id, object] in self.state.novel_objects()]

class CartPoleObservation(HydraObservation):
    ''' An object that represents an observation in the cartpole domain.'''

    def __init__(self):
        self.states = [] # An SBState
        self.actions = []  # an SBAction
        self.rewards = [] # The reward obtained from performing an action

    def get_initial_state(self):
        return self.states[0]

    def get_pddl_states_in_trace(self, meta_model: CartPoleMetaModel = CartPoleMetaModel()) -> list: # TODO: Refactor and move this to the meta model?
        ''' Returns a sequence of PDDL states that are the observed intermediate states '''
        observed_state_seq = []
        for state in self.states:
            pddl = meta_model.create_pddl_state(state)
            observed_state_seq.append(pddl)
        return observed_state_seq

    def get_pddl_plan(self, meta_model: CartPoleMetaModel = CartPoleMetaModel):
        ''' Returns a PDDL+ plan object with a single action that is the action that was performed '''
        pddl_plan = PddlPlusPlan()
        previous_action_name = "move_cart_right dummy_obj" # TODO: Better to get the default side from the meta model, but also better to discuss design
        for ix in range(len(self.actions)):
            timed_action = meta_model.create_timed_action(self.actions[ix], ix)
            if timed_action.action_name!=previous_action_name:
                pddl_plan.append(timed_action)
                previous_action_name = timed_action.action_name
        return pddl_plan
