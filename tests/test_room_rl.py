import worlds.room_world as rm


def test_room_rl(): # this works
    world = rm.RoomWorld()
    assert len(world.actions) == 14
    state,reward = world.act(world.actions[0])
    assert state.room == 2 # failed action
    assert reward == -1
    assert len(world.available_actions()) == 3
    state,reward = world.act(world.actions[10])
    assert state.room == 3 # failed action
    assert reward == 0



import numpy as np


def test_room_sarsa_learner():
    np.random.seed(0)
    world = rm.RoomWorld()  # initialize world
    states = [rm.RoomState(i) for i in range(0,7)]
    Q = np.array(np.zeros(shape=(8, 8))) #Q[1][2] is the predicted reward for moving from 1 to 2


    # learning parameter

    alpha = 0.81
    gamma = 0.8

    def choose_action(epsilon = 0.9):
        if np.random.uniform(0, 1) < epsilon:
            return np.random.choice(world.available_actions())
        else:
            max_policy = max(Q[world.current_state.room, :])
            max_actions = [act for act in world.available_actions() if
                                       Q[world.current_state.room][act.to_id] == max_policy]
            if len(max_actions) == 0:
                import pdb
                pdb.set_trace()
            else:
                return np.random.choice(max_actions)

    def update(state, state2, reward, action, action2):
        predict = Q[state.room, action.to_id]
        target = reward + gamma * Q[state2.room, action2.to_id]
        Q[state.room, action.to_id] = Q[state.room, action.to_id] + alpha * (target - predict)

        # Training
    total_episodes = 10000
    max_steps = 20
    for i in range(total_episodes):
#        print("training: "+str(i))
        t = 0
        world = rm.RoomWorld() #reset env
        state1 = world.current_state
        action1 = choose_action()
        while t < max_steps:
            state2, reward = world.act(action1)
            action2 = choose_action()
            update(state1,state2,reward,action1,action2)
            state1 = state2
            action1 = action2
            t+=1
            if reward == 100:
                break # reached goal
#        if i % 100 == 0:
#            print('Q Mean: ', Q.mean())

    print("Trained Q matrix:")
    print(Q / np.max(Q) * 100)

    world = rm.RoomWorld()  # reset env
    route = []
    t = 0
    while (world.current_state.room != 7 and t < max_steps):
        route.append(world.current_state.room)
        world.act(choose_action(0))
        t+=1
    route.append(world.current_state.room)
    print(route)
    assert route == [2,5,6,7]
