from typing import List

from worlds.science_birds import SBAction, SBState
from agent.consistency.episode_log import HydraEpisodeLog
from agent.planning.pddl_plus import PddlPlusPlan, PddlPlusState
from agent.planning.sb_meta_model import ScienceBirdsMetaModel


class SBEpisodeLog(HydraEpisodeLog):
    ''' An object that represents an observation of the SB game '''
    state: SBState
    action: SBAction
    intermediate_states: List[SBState]
    reward: float

    def __init__(self):
        self.state = None  # An SBState
        self.action = None  # an SBAction
        self.intermediate_states = None  # The  sequence of intermediates states observed after doing the action
        self.reward = 0.0  # The reward obtained from performing an action

    def get_initial_state(self) -> SBState:
        return self.state

    def get_pddl_states_in_trace(self,
                                 meta_model: ScienceBirdsMetaModel) -> List[PddlPlusState]:  # TODO: Refactor and move this to the meta model?
        """ Returns a sequence of PDDL states that are the observed intermediate states """
        observed_state_seq = []
        for intermediate_state in self.intermediate_states:
            observed_state_seq.append(meta_model.create_pddl_state(intermediate_state))
        return observed_state_seq

    def get_pddl_plan(self, meta_model: ScienceBirdsMetaModel = ScienceBirdsMetaModel) -> PddlPlusPlan:
        """ Returns a PDDL+ plan object with a single action that is the action that was performed """
        pddl_plan = PddlPlusPlan()
        pddl_plan.append(meta_model.create_timed_action(self.action, self.state))
        return pddl_plan

    def hasUnknownObj(self) -> bool:
        if self.state.novel_objects():
            return True
        else:
            return False

    def get_novel_object_ids(self) -> List[str]:
        ''' Return a list of novel object ids '''
        return [object_id for [object_id, object] in self.state.novel_objects()]