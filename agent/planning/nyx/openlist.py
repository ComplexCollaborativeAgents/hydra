from collections import deque

from agent.planning.nyx.abstract_heuristic import AbstractHeuristic


class OpenList:
    """
    Base class for open lists. Holds search nodes yet to be expanded, implements simple fifo (BFS).
    """

    def __init__(self):
        self.nodes = deque()

    def push(self, node):
        """
        Adds a node to the fringe. A fringe can choose not to add a node if it knows it's a dead end (e.g. if the
        heuristic value is infinity and the heuristic is sound).
        """
        self.nodes.append(node)

    def pop(self):
        """
        Removes and returns the next node.
        """
        assert bool(self), 'Can not pop empty list!'
        return self.nodes.popleft()

    def top(self):
        """
        Return the next node in the fringe without removing it.
        """
        assert bool(self), 'Can not get top of empty list!'
        return self.nodes[0]

    def clear(self):
        """
        Removes all nodes from the fringe.
        """
        self.nodes.clear()

    def __bool__(self):
        return bool(self.nodes)


class DFSList(OpenList):
    """
    Uses lifo, for DFS search
    """

    def __init__(self):
        OpenList.__init__(self)
        self.nodes = []

    def push(self, node):
        self.nodes.append(node)

    def pop(self):
        assert bool(self), 'Can not pop empty list!'
        return self.nodes.pop()

    def top(self):
        assert bool(self), 'Can not get top of empty list!'
        return self.nodes[-1]

    def clear(self):
        self.nodes.clear()

    def __bool__(self):
        return bool(self.nodes)


class PriorityList(OpenList):
    """
    Sorts open list by heuristic value. Parameter 'astar' controls Best-first vs A* search
    """

    def __init__(self, Astar=True):
        OpenList.__init__(self)
        self.nodes = dict()
        self.min_val = float("inf")
        self.astar = Astar
        # Keep track of min value so we don't have to search for it each time we want the next node.
        # self.nodes will be dict of fifos.

    def push(self, node):
        if self.astar:
            cost = node.h + node.g
            if not self.nodes.get(cost):
                self.nodes[cost] = OpenList()
            if cost < self.min_val:
                self.min_val = cost
            self.nodes[cost].push(node)
        else:
            if not self.nodes.get(node.h):
                self.nodes[node.h] = OpenList()
                if node.h < self.min_val:
                    self.min_val = node.h
            self.nodes[node.h].push(node)

    def top(self):
        assert bool(self), 'Can not get top of empty list!'
        return self.nodes[self.min_val].top()

    def pop(self):
        assert bool(self), 'Can not pop empty list!'
        next_node = self.nodes[self.min_val].pop()
        if not self.nodes[self.min_val]:
            del self.nodes[self.min_val]  # deleting the element takes time, but allows __bool__ to indicate emptiness.
            if self.nodes:
                self.min_val = min(self.nodes.keys())
            else:
                self.min_val = float("inf")
        return next_node

    def clear(self):
        self.nodes.clear()
        self.min_val = float("inf")

    def __bool__(self):
        return bool(self.nodes)


class PreferredList(OpenList):
    """
    Only accepts preferred nodes
    """

    def __init__(self, parent: OpenList, heuristic: AbstractHeuristic):
        OpenList.__init__(self)
        self.heuristic = heuristic
        self.parent = parent

    def push(self, node):
        if self.heuristic.is_preferred(node):
            self.parent.push(node)

    def top(self):
        assert bool(self), 'Can not get top of empty list!'
        return self.parent.top()

    def pop(self):
        assert bool(self), 'Can not pop empty list!'
        return self.parent.pop()

    def clear(self):
        self.parent.clear()

    def __bool__(self):
        return bool(self.parent)


class AlternatingList(OpenList):
    """
    Alternates between several lists, for multi-heuristic or multi-goal search.
    See https://ai.dmi.unibas.ch/papers/helmert-jair06.pdf, SMHA*, others.
    """

    def __init__(self, lists: list):
        OpenList.__init__(self)
        assert lists
        self.lists = lists
        self.next_index = 0

    def push(self, node):
        for open_list in self.lists:
            open_list.push(node)

    def top(self):
        assert bool(self), 'Can not get top of empty list!'
        while not self.lists[self.next_index]:
            # scroll to next non-empty list
            self.next_index = (self.next_index + 1) % len(self.lists)
        return self.lists[self.next_index].top()

    def pop(self):
        assert bool(self), 'Can not pop empty list!'
        while not self.lists[self.next_index]:
            # scroll to next non-empty list
            self.next_index = (self.next_index + 1) % len(self.lists)
        next_node = self.lists[self.next_index].pop()
        self.next_index = (self.next_index + 1) % len(self.lists)
        return next_node

    def clear(self):
        self.lists.clear()

    def __bool__(self):
        return any(self.lists)


def BestFirstList():
    return PriorityList(astar=False)


def AstarList():
    return PriorityList(astar=True)


BFSList = OpenList
