
import worlds.science_birds as sb
import pytest
import sys
from os import path
import settings

@pytest.mark.skipif(settings.HEADLESS==True)
def test_science_birds():
    print("starting")
    env = sb.ScienceBirds()
    state = env.get_current_state()
    print(state.objects)
    assert(len(state.objects) == 109)

    ref_point = env.sb_client.tp.get_reference_point(env.cur_sling)
    action = sb.SBAction(ref_point.X + 25,ref_point.Y + 25,50) # no idea what the scale of these should be
    state,reward,done = env.act(action)
    assert isinstance(state,sb.SBState)
    assert reward == 0
    assert not done
    env.kill()

def test_state_serialization():
    # state = SBState.serialize_current_state(path.join(settings.ROOT_PATH, 'data', 'science_birds', 'serialized_levels', 'level-00.p'))
    loaded_serialized_state = sb.SBState.load_from_serialized_state(path.join(settings.ROOT_PATH, 'data', 'science_birds', 'serialized_levels', 'level-00.p'))
    assert isinstance(loaded_serialized_state, sb.SBState)
    # assert state.objects == loaded_serialized_state.objects