import logging
from typing import List

from agent.consistency.episode_log import HydraEpisodeLog
from agent.planning.pddl_plus import PddlPlusPlan, PddlPlusState
from agent.planning.polycraft_meta_model import PolycraftMetaModel


class PolycraftEpisodeLog(HydraEpisodeLog):
    """ An object that represents an observation of the SB game """

    def __init__(self, meta_model):
        self.states = []  # A sequence of polycraft states
        self.actions = []  # A sequence of polycraft actions
        self.rewards = []  # The reward obtained from performing each action
        self.meta_model = meta_model
        self.time_so_far = 0.0
        self.pddl_states = []

    def get_initial_state(self):
        return self.states[0]  # self.meta_model.create_pddl_state(self.states[0])

    def get_pddl_states_in_trace(self,
                                 meta_model: PolycraftMetaModel) -> List[PddlPlusState]:
        # TODO: Refactor and move this to the meta model?
        """ Returns a sequence of PDDL states that are the observed intermediate states """
        if len(self.pddl_states) < len(self.states):
            self.pddl_states = [meta_model.create_pddl_state(state) for state in self.states]
        return self.pddl_states

    def get_pddl_plan(self, meta_model: PolycraftMetaModel):
        """ Returns a PDDL+ plan object with the actions we performed """
        return PddlPlusPlan(self.actions)

    def print(self):
        for i, state in enumerate(self.states):
            print(f'State[{i}] {str(state)}')
            print(f'Action[{i}] {str(self.actions[i])}')
