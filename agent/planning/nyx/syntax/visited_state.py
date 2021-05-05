# -----------------------------------------------
# Wrapper for the State class with secondary comparison functions
# Used for duplicate checks.
# -----------------------------------------------

class VisitedState():

    def __init__(self, state):
        self.state = state

    def __eq__(self, other):
        return other and self.state.time == other.state.time and self.state.state_vars == other.state.state_vars

    def __hash__(self):
        return hash((self.state.time, str(self.state.state_vars)))

    def __lt__(self, other):
        return other and self.state.time < other.state.time and self.state.state_vars < other.state.state_vars

    def __gt__(self, other):
        return other and self.state.time > other.state.time and self.state.state_vars > other.state.state_vars

    def __ge__(self, other):
        return other and self.state.time >= other.state.time and self.state.state_vars >= other.state.state_vars

    def __le__(self, other):
        return other and self.state.time <= other.state.time and self.state.state_vars <= other.state.state_vars

    def __ne__(self, other):
        return not self.__eq__(other)


    # def __eq__(self, other):
    #     return (self.state.h + self.state.g) == (other.state.h + other.state.g)
    #
    # def __hash__(self):
    #     return hash(self.state.h + self.state.g)
    #
    # def __lt__(self, other):
    #     return (self.state.h + self.state.g) < (other.state.h + other.state.g)
    #
    # def __gt__(self, other):
    #     return (self.state.h + self.state.g) > (other.state.h + other.state.g)
    #
    # def __ge__(self, other):
    #     return (self.state.h + self.state.g) >= (other.state.h + other.state.g)
    #
    # def __le__(self, other):
    #     return (self.state.h + self.state.g) <= (other.state.h + other.state.g)
    #
    # def __ne__(self, other):
    #     return (self.state.h + self.state.g) != (other.state.h + other.state.g)