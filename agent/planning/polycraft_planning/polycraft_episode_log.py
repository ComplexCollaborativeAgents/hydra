import logging
from typing import List

from worlds.polycraft_world import PolycraftAction, PolycraftState
from agent.consistency.observation import HydraEpisodeLog
from agent.planning.pddl_plus import PddlPlusPlan, PddlPlusState
from agent.planning.polycraft_meta_model import PolycraftMetaModel


class PolycraftEpisodeLog(HydraEpisodeLog):
    """An object that represents an observation of the SB game 
    """

    states: List[PolycraftState]    # A sequence of polycraft states
    actions: List[PolycraftAction]  # A sequence of polycraft actions
    rewards: List[float]            # The reward obtained from performing each action
    meta_model: PolycraftMetaModel
    time_so_far: float              # The time since an episode has started

    def __init__(self, meta_model: PolycraftMetaModel):
        self.states = []  
        self.actions = [] 
        self.rewards = []
        self.meta_model = meta_model
        self.time_so_far = 0.0
        self.pddl_states = []

    def get_initial_state(self) -> PolycraftState:
        """Get the initial state of the episode log

        Returns:
            PolycraftState: Polycraft World state object
        """
        return self.states[0]

    def get_pddl_states_in_trace(self,
                                 meta_model: PolycraftMetaModel) -> List[PddlPlusState]:
        # TODO: Refactor and move this to the meta model?
        """Returns a sequence of PDDL states that are the observed intermediate states

        Args:
            meta_model (PolycraftMetaModel): Meta Model that contains the intermediate states

        Returns:
            List[PddlPlusState]: List of pddl states
        """
        if len(self.pddl_states) < len(self.states):
            self.pddl_states = [meta_model.create_pddl_state(state) for state in self.states]
        return self.pddl_states

    def get_pddl_plan(self, meta_model: PolycraftMetaModel) -> PddlPlusPlan:
        """ Returns a PDDL+ plan object with the actions we performed """
        return PddlPlusPlan(self.actions)

    def print(self):
        for i, state in enumerate(self.states):
            print(f'State[{i}] {str(state)}')
            print(f'Action[{i}] {str(self.actions[i])}')
