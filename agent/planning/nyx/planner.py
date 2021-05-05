#!/usr/bin/env python
# Four spaces as indentation [no tabs]
import bisect
from hmac import new

from agent.planning.nyx import heuristic_functions
from agent.planning.nyx.PDDL import PDDL_Parser
import agent.planning.nyx.syntax.constants as constants
import time, copy
from agent.planning.nyx.syntax.visited_state import VisitedState
from agent.planning.nyx.syntax.state import State

class Planner:

    #-----------------------------------------------
    # Solve
    #-----------------------------------------------

    initial_state = None
    reached_goal_state = None
    explored_states = 0
    # total_visited = 0
    queue = []
    visited_hashmap = {}


    def solve(self, domain, problem):

        start_solve_time = time.time()
        # Parser
        parser = PDDL_Parser(domain, problem)
        # Parsed data
        state = parser.init_state
        self.initial_state = parser.init_state

        print("\t* model parse time: " + str("{:5.4f}".format(time.time()-start_solve_time)) + "s")

        # Do nothing
        if state.is_goal(parser.goals):
            return []

        # Search
        self.visited_hashmap[hash(VisitedState(state))] = VisitedState(state)
        self.queue = [state]
        while self.queue:
            state = self.queue.pop(0)
            for aa in state.get_applicable_happenings(parser.grounded_actions):
                if aa == constants.TIME_PASSING_ACTION:
                    new_state = copy.deepcopy(state)
                    for hp in state.get_applicable_happenings(parser.grounded_events)+state.get_applicable_happenings(parser.grounded_processes):
                        new_state = new_state.apply_happening(hp)
                    new_state.set_time(round(round(state.time,constants.NUMBER_PRECISION) + round(constants.DELTA_T,constants.NUMBER_PRECISION),constants.NUMBER_PRECISION))
                    new_state.predecessor_hashed = hash(VisitedState(state))
                    new_state.predecessor_action = aa
                else:
                    new_state = state.apply_happening(aa)
                self.explored_states += 1
                if hash(VisitedState(new_state)) not in self.visited_hashmap and new_state.time <= constants.TIME_HORIZON:
                    if new_state.is_goal(parser.goals):
                        self.reached_goal_state = new_state
                        # self.total_visited = len(self.visited_hashmap)
                        return self.reached_goal_state
                    self.visited_hashmap[hash(VisitedState(new_state))] = VisitedState(new_state)
                    self.enqueue_state(new_state)

                if self.explored_states % constants.PRINT_INFO == 0:
                    # visi = len(self.visited_hashmap)
                    time_checkpoint = time.time() - start_solve_time
                    print('[' + str("{:6.2f}".format(time_checkpoint)) + '] ==> states explored: ' + str(self.explored_states))
                    print('\t\t\t' + str(round(self.explored_states / time_checkpoint,2)) + ' states/sec')

                if constants.PRINT_ALL_STATES:
                    print(new_state)

        return None

    def enqueue_state(self, n_state):
        if constants.SEARCH_BFS:
            self.queue.append(n_state)
        elif constants.SEARCH_DFS:
            self.queue.insert(0, n_state)
        elif constants.SEARCH_GBFS:
            n_state.set_h_heuristic(heuristic_functions.heuristic_function(n_state))
            ''' changing enqueue to bisect.insort ==> needs performance comparison '''
            # self.queue.insert(0, n_state)
            # self.queue = sorted(self.queue, key=lambda elem: (elem.h))

            bisect.insort(self.queue, n_state)

            # self.queue.insert(bisect.bisect_left(self.queue, n_state), n_state)

        elif constants.SEARCH_ASTAR:
            n_state.set_h_heuristic(heuristic_functions.heuristic_function(n_state))
            self.queue.insert(0, n_state)
            self.queue = sorted(self.queue, key=lambda elem: (elem.h+elem.g))

    def get_trajectory(self, sstate: State):
        plan = []
        curr_v_state = VisitedState(sstate)

        while curr_v_state.state.predecessor_action is not None:
            plan.insert(0,(curr_v_state.state.predecessor_action, copy.deepcopy(curr_v_state.state)))
            curr_v_state = self.visited_hashmap[curr_v_state.state.predecessor_hashed]
        return plan
