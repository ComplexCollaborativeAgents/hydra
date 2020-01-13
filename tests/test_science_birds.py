
import worlds.science_birds as sb

@pytest.mark.skip(reason='No headless launching of Science Birds yet.')
def test_science_birds():
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
