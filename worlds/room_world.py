import numpy as np
from utils.state import State, Action, World


class RoomAction(Action):
    """Takes the action to move to a room, even if not possible"""
    def __init__(self,from_id, to_id):
        self.to_id = to_id
        self.from_id = from_id

    def get_rl_id(self):
        return self.to_id


class RoomState(State):
    """State of room world"""

    def __init__(self,room):
        self.room = room

    def get_rl_id(self):
        return self.room

class RoomWorld(World):
    """
    Simple room navigation taken form
    https://amunategui.github.io/reinforcement-learning/index.html
    """
# map cell to cell, add circular cell to goal point
    def __init__(self):
        self.actions = []
        points_list = [(0,1), (1,5), (5,6), (5,4), (1,2), (2,3), (2,5),(6,7)]
        self.history = []
        self.goal = 7
        self.current_state = RoomState(2)

        MATRIX_SIZE = 8
        self.Reward = np.array(np.ones(shape=(MATRIX_SIZE, MATRIX_SIZE)))

        self.Reward *= -1

        for point in points_list:
            self.actions.append(RoomAction(point[0],point[1]))
            self.actions.append(RoomAction(point[1],point[0]))

            if point[1] == self.goal:
                self.Reward[point] = 100
            else:
                self.Reward[point] = 0

            if point[0] == self.goal:
                self.Reward[point[::-1]] = 100
            else:
                # reverse of point
                self.Reward[point[::-1]] = 0
        self.actions.append(RoomAction(self.goal,self.goal))
        self.Reward[self.goal, self.goal] = 100

    def available_actions(self,state=None):
        """This is a discrete set, probably doesn't work for birds"""
        state = state if state else self.current_state
        return [act for act in self.actions if act.from_id == state.room]

    def get_current_state(self):
        return self.current_state

    def act(self,action):
        '''returns the new current state and reward'''
        self.history.append(action.get_rl_id()) # a better specifier would make sense.
        if (self.current_state.room == action.from_id):
            reward = self.Reward[self.current_state.room][action.to_id]
            self.current_state = RoomState(action.to_id)
            return self.current_state, reward, reward == 100
        else:
            assert None
            return self.current_state,-1, False




