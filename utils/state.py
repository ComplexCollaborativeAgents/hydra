import itertools

class State():
    """This is the most abstract state class that captures the current state of the game"""

    newid = itertools.count()
    def __init__(self):
        self.id = next(State.newid)

class Action():
    """
    This class defines actions and will be specialized in worlds
    """
    newid = itertools.count()

    def __init__(self):
        self.id = next(Action.newid)

class World():
    def __init__(self):
        self.actions = []
        self.current_state = None
        self.Reward = None

    def available_actions(self):
        pass

    def act(self,action):
        pass

