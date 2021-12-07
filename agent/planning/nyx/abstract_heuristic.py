from typing import List


class AbstractHeuristic:
    """
    Abstract class for heuristics - implements null heuristic (always returns 0)
    """
    @classmethod
    def notify_initial_state(cls, node):
        """
        Sets up heuristic, also evaluates the initial state and returns it's value. If the heuristic wants to do any
        preprocessing, this is the place.
        """
        return cls.evaluate(node)

    @classmethod
    def notify_expanded(cls, node):
        """
        Notifies the heuristic that this node is about to be expanded (useful for some heuristics).
        May return False to indicate a dead-end, in which case expansion should be aborted.
        :return False if this node is a dead end.
        """
        return True

    @classmethod
    def evaluate(cls, node):
        """
        Evaluates the search node and returns it's heuristic value (updating the node with the value).
        It is very important to return the newly calculated value, as the node might store some other value internally
        in multi-goal search settings.
        """
        node.h = 0
        return 0

    @classmethod
    def is_preferred(cls, node):
        """
        Is this a preferred search node? (used by some heuristics)
        """
        return False


class ZeroHeuristic(AbstractHeuristic):
    pass


class CompositeHeuristic(AbstractHeuristic):
    """
    Uses a fast heuristic when opening (generating) nodes and a slow but more informative one when expanding.
    See https://ai.dmi.unibas.ch/papers/helmert-jair06.pdf for good rational of doing this.
    """
    def __init__(self, fast_h: AbstractHeuristic, slow_h: AbstractHeuristic):
        self.fast = fast_h
        self.slow = slow_h

    def notify_expanded(self, node):
        return self.slow.evaluate(node)

    def evaluate(self, node):
        self.fast.evaluate(node)
        return node.h

    def is_preferred(self, node):
        return self.fast.is_preferred(node) or self.slow.is_preferred(node)


class DifferenceHeuristic(AbstractHeuristic):
    """
    Hamming distance for variable based planning problems: how many variables have values different from the goal.
    """

    def __init__(self, goal_vals: dict):
        self.goals = goal_vals

    def evaluate(self, node):
        heuristic = 0
        for var, val in self.goals.items():
            if node.state[var] != val:
                heuristic += 1
        node.h = heuristic
        return heuristic


class HeuristicSum(AbstractHeuristic):
    """
    Returns the sum of several heuristics
    """
    def __init__(self, h_list: List[AbstractHeuristic]):
        self.h_list = h_list

    def notify_initial_state(self, node):
        for h in self.h_list:
            h.notify_initial_state(node)

    def notify_expanded(self, node):
        for h in self.h_list:
            h.notify_expanded(node)

    def evaluate(self, node):
        return sum([h.evaluate(node) for h in self.h_list])

    def is_preferred(self, node):
        return any([h.is_preferred(node) for h in self.h_list])
