import worlds.room_world as rm

def test_room_rl():
    world = rm.RoomWorld()
    assert len(world.actions) == 7
    state,reward = world.act(world.actions[0])
    assert state.room == 1 # failed action
    assert reward == -1
    state,reward = world.act(world.actions[1])
    assert state.room == 5 # failed action
    assert reward == 0
