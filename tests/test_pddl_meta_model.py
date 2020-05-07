import worlds.science_birds
from os import path
from agent.planning.pddl_meta_model import MetaModel
import settings

''' Checks if the meta_model generates the same PddlPlusProblem as SBState.translate...()'''
def test_translate_sb_state_to_pddl_problem():
    SB_STATE_FILE_PATH = path.join(settings.ROOT_PATH, 'data', 'science_birds', 'serialized_levels', 'level-01','dx_test_0.p')
    state = worlds.science_birds.SBState.load_from_serialized_state(SB_STATE_FILE_PATH)

    pddl_problem1 = state.translate_state_to_pddl()
    meta_model = MetaModel()
    pddl_problem2 = meta_model.translate_sb_state_to_pddl_problem(state)

    assert pddl_problem1==pddl_problem2

def test_translate_sb_state_to_pddl_state():
    SB_STATE_FILE_PATH = path.join(settings.ROOT_PATH, 'data', 'science_birds', 'serialized_levels', 'level-01','dx_test_3.p')
    state = worlds.science_birds.SBState.load_from_serialized_state(SB_STATE_FILE_PATH)

    pddl_state1 = state.translate_intermediate_state_to_pddl_state()
    meta_model = MetaModel()
    pddl_state2 = meta_model.translate_sb_state_to_pddl_state(state)

    assert pddl_state1==pddl_state2
