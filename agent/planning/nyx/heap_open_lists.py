import heapq

from agent.planning.nyx.openlist import OpenList


class HeapPriorityList(OpenList):
    """
    An open list based on a heap.
    """
    # NOTE(Yoni): This was profiled in comparison to the other open lists, and found to be equivalent for the problems
    #  in the SAIL-ON project. 
    def __init__(self, Astar=True):
        super().__init__()
        self.astar = Astar
        self.nodes = []

    def push(self, node):
        if self.astar:
            heapq.heappush(self.nodes, (node.g + node.h, node))
        else:
            heapq.heappush(self.nodes, (node.h, node))

    def pop(self):
        assert bool(self), 'Can not pop empty list!'
        return heapq.heappop(self.nodes)[1]

    def top(self):
        assert bool(self), 'Can not pop empty list!'
        return self.nodes[0][1]
