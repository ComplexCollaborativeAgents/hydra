# Simple Sarsa-Learning
from worlds.room_world import RoomWorld
import numpy as np

class SarsaLearner():

    def __init__(self,env):
        assert(isinstance(env,RoomWorld)) # Q[1][2] is the predicted reward for moving from 1 to 2
        self.Q = np.array(np.zeros(shape=(8, 8)))

        self.env = env # need to be able to check available actions
        self.alpha = 0.81
        self.gamma = 0.8
        self.epsilon = .9
        
    def choose_action(self,state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.env.available_actions(state))
        else:
            max_policy = max(self.Q[self.env.current_state.room, :])
            max_actions = [act for act in self.env.available_actions(state) if
                                       self.Q[self.env.current_state.room][act.to_id] == max_policy]
            print(max_policy)
            return np.random.choice(max_actions)

    def update(self,state, state2, reward, action, action2):
        predict = self.Q[state.get_rl_id(), action.get_rl_id()]
        target = reward + self.gamma * self.Q[state2.get_rl_id(), action2.get_rl_id()]
        self.Q[state.get_rl_id(), action.get_rl_id()] = self.Q[state.get_rl_id(), action.get_rl_id()] + self.alpha * (target - predict)


    def act_and_learn(self,state1):
        action1 = self.choose_action(state1)
        state2, reward, done = self.env.act(action1)
        action2 = self.choose_action(state2)
        self.update(state1, state2, reward, action1, action2)
        return state2, reward, done




