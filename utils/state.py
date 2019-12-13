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
    pass

class World():
    pass
