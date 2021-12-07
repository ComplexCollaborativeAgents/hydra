import heapq

from agent.planning.nyx.openlist import OpenList


class HeapPriorityList(OpenList):
    """
    An open list based on a heap.
    """
    # Push and pop are taking 4216 ms (0.4%) and 13515 ms (1.2%) of runtime.
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
