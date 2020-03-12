import worlds.room_world as rm
import agent.hydra_agent as ha
import numpy as np
import pytest

pytest.skip("Hydra only supports science birds now")
def test_room_rl(): # ensures the environment works
    world = rm.RoomWorld()
    assert len(world.actions) == 17
    assert len(world.available_actions()) == 3
    state,reward, done = world.act(world.available_actions()[0])
    assert state.room == 1
    assert reward == 0
    assert not done
    with pytest.raises(AssertionError): # illegal action
        state,reward,done = world.act(world.actions[0])


pytest.skip("Hydra only supports science birds now")
def test_room_sarsa_learner():
    np.random.seed(0)
    env = rm.RoomWorld()  # initialize world
    agent = ha.HydraAgent(env)

    # Training
    total_episodes = 100
    max_steps = 20
    for i in range(total_episodes):
        agent.set_env(rm.RoomWorld())  # reset env
        agent.main_loop()  # complete the level, maybe this should be part of the planner as well

    # Testing
    agent.set_env(rm.RoomWorld())  # reset env
    agent.rl.epsilon = 0  # should just do policy and not doing learning
    agent.main_loop()
    print("Plan: " + str(agent.env.history))
    assert (agent.env.history == [5, 6, 7])
