import worlds.science_birds
from os import path
from agent.planning.pddl_meta_model import MetaModel
import settings

''' Checks if the meta_model generates the same PddlPlusProblem as SBState.translate...()'''
def test_translate_sb_state_to_pddl_problem():
    SB_STATE_FILE_PATH = path.join(settings.ROOT_PATH, 'data', 'science_birds', 'serialized_levels', 'level-01','dx_test_0.p')
    state = worlds.science_birds.SBState.load_from_serialized_state(SB_STATE_FILE_PATH)

    (prob1, prob_simplified1, _) = state.translate_state_to_pddl()
    meta_model = MetaModel()
    prob2 = meta_model.create_pddl_problem(state)
    prob_simplified2 =meta_model.create_simplified_problem(prob2)

    assert prob1==prob2
    assert prob_simplified1 == prob_simplified2

