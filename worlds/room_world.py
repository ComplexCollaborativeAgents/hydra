import numpy as np
from utils.state import State, Action, World


class RoomAction(Action):
    """Takes the action to move to a room, even if not possible"""
    def __init__(self,from_id, to_id):
        self.to_id = to_id
        self.from_id = from_id


class RoomState(State):
    """State of room world"""

    def __init__(self,room):
        self.room = room

class RoomWorld(World):
    """
    Simple room navigation taken form
    https://amunategui.github.io/reinforcement-learning/index.html
    """
# map cell to cell, add circular cell to goal point
    def __init__(self):
        self.actions = []
        points_list = [(0,1), (1,5), (5,6), (5,4), (1,2), (2,3), (2,7)]
        self.goal = 7
        self.current_state = RoomState(1)
        MATRIX_SIZE = 8
        self.Reward = np.array(np.ones(shape=(MATRIX_SIZE, MATRIX_SIZE)))

        self.Reward *= -1

        for point in points_list:
            self.actions.append(RoomAction(point[0],point[1]))
            print(point)
            if point[1] == self.goal:
                self.Reward[point] = 100
            else:
                self.Reward[point] = 0

            if point[0] == self.goal:
                self.Reward[point[::-1]] = 100
            else:
                # reverse of point
                self.Reward[point[::-1]] = 0
        self.Reward[self.goal, self.goal] = 100

    def act(self,action):
        '''returns the new current state and reward'''
        reward = self.Reward[self.current_state.room][action.to_id]
        if (self.current_state.room == action.from_id):
            self.current_state = RoomState(action.to_id)
        return self.current_state,reward




