import os.path

from agent.planning.pddl_meta_model import *
from agent.planning.pddl_plus import *
from worlds.science_birds import *
import pytest

DATA_DIR = path.join(settings.ROOT_PATH, 'data')
DATA_TESTS_DIR = path.join(DATA_DIR, 'science_birds', 'tests')

''' Check creating SBAction and PDDL+ timed action  '''
@pytest.mark.skip("obs_test_meta_model.p was created by an older version of science birds and therefore is not consistent with current system")
def test_action_creation():
    observation = pickle.load(open(os.path.join(DATA_TESTS_DIR, "obs_test_meta_model.p"), "rb"))
    meta_model = MetaModel()
    pddl_state = PddlPlusState(meta_model.create_pddl_problem(observation.state).init)

    # Create action
    t = 1.025
    angle = MetaModel.action_time_to_angle(t, pddl_state)
    sb_action = meta_model.create_sb_action(TimedAction(meta_model.TWANG_ACTION, t), observation.state)

    # Compute angle and time from SB Action
    timed_action = meta_model.create_timed_action(sb_action, observation.state)

    assert abs(timed_action.start_at - t)<0.5

    action_angle = meta_model.action_time_to_angle(timed_action.start_at,pddl_state)
    assert abs(action_angle - angle) < 0.5

    # Re-create the SB Action, make sure it is almost the same
    sb_action2 = meta_model.create_sb_action(TimedAction(meta_model.TWANG_ACTION, timed_action.start_at), observation.state)
    assert abs(sb_action.ref_x - sb_action2.ref_x) < 1.5
    assert abs(sb_action.ref_y - sb_action2.ref_y) < 1.5
    assert abs(sb_action.dx - sb_action2.dx) < 1.5
    assert abs(sb_action.dy - sb_action2.dy) < 1.5

''' Check conversion from angle to time is correct '''
@pytest.mark.skip("Have not migrated to 0.3.6 yet")
def test_action_angle_conversion():
    observation = pickle.load(open(os.path.join(DATA_TESTS_DIR, "obs_test_meta_model.p"), "rb"))

    meta_model = MetaModel()
    pddl_state = PddlPlusState(meta_model.create_pddl_problem(observation.state).init)

    for t_index in range(1, 4000):
        t = t_index/1000

        # Test angle to time conversion
        action_angle = MetaModel.action_time_to_angle(t, pddl_state)
        derived_t = MetaModel.angle_to_action_time(action_angle, pddl_state)
        assert abs(t-derived_t)<0.05



''' Check the bird x computation when creating a problem and when creating an intermediate state '''
def test_bird_x_computation():
    observation = pickle.load(open(os.path.join(DATA_TESTS_DIR, "bad_observation.p"), "rb"))
    meta_model = MetaModel()

    state2 = PddlPlusState(meta_model.create_pddl_problem(observation.state).init)
    state1 = meta_model.create_pddl_state(observation.state)

    for bird in state1.get_birds():
        fluent = ('x_bird', bird)
        birdx1 = state1.numeric_fluents[fluent]
        birdx2 = state2.numeric_fluents[fluent]
        assert birdx1==birdx2
