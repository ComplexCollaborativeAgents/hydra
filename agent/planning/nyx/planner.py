#!/usr/bin/env python
# Four spaces as indentation [no tabs]

import copy
import time

import agent.planning.nyx.semantic_attachments as semantic_attachments
import agent.planning.nyx.syntax.constants as constants
from agent.planning.nyx.PDDL import PDDL_Parser
from agent.planning.nyx.heap_open_lists import HeapPriorityList
from agent.planning.nyx.heuristic_functions import get_heuristic_function
from agent.planning.nyx.openlist import BFSList, DFSList
from agent.planning.nyx.syntax.state import State
from agent.planning.nyx.syntax.visited_state import VisitedState


class Planner:

    #-----------------------------------------------
    # Solve
    #-----------------------------------------------

    def __init__(self):
        self.initial_state = None
        self.reached_goal_states = list()
        self.explored_states = 0
        # self.total_visited = 0
        self.visited_hashmap = {}
        self.heuristic = None
        self.queue = self._get_open_list()  # TODO get this parameter in some normal way

    def solve(self, domain, problem):

        start_solve_time = time.time()
        # Parser
        parser = PDDL_Parser(domain, problem)
        grounded_instance = parser.grounded_instance
        self.heuristic = get_heuristic_function(constants.CUSTOM_HEURISTIC_ID, groundedPPDL=grounded_instance)  # TODO get this parameter in some normal way
        # Parsed data
        state = grounded_instance.init_state
        self.initial_state = grounded_instance.init_state
        self.heuristic.notify_initial_state(state)

        print("\t* model parse time: " + str("{:5.4f}".format(time.time() - start_solve_time)) + "s")

        # Do nothing
        if grounded_instance.goals(state, constants):
            return []

        # Search
        self.visited_hashmap[hash(VisitedState(state))] = VisitedState(state)
        self.queue.push(state)
        while self.queue:
            state = self.queue.pop()
            from_state = VisitedState(state)
            time_passed = round(state.time + constants.DELTA_T, constants.NUMBER_PRECISION)
            for aa in grounded_instance.actions.get_applicable(state):
                new_state = None
                if aa == constants.TIME_PASSING_ACTION:
                    new_state = state
                    # new_state = State(t=round(state.time, constants.NUMBER_PRECISION), g=state.g + 1, predecessor=state, predecessor_action=aa)
                    # first check for triggered events, then semantic attachment methods, followed by processes, and events again if '-dblevent' flag is true
                    happenings_list = grounded_instance.events.get_applicable(state)
                    for hp in happenings_list:
                        new_state = new_state.apply_happening(hp, from_state=from_state, create_new_state=new_state is state)

                    # check whether any semantic attachment processes are active, if applicable
                    if constants.SEMANTIC_ATTACHMENT:
                        if new_state is state:
                            new_state = State(t=round(state.time, constants.NUMBER_PRECISION), g=state.g + 1, predecessor=state, predecessor_action=aa)
                        new_state = semantic_attachments.semantic_attachment.external_function(new_state)

                    # first check for triggered events, followed by processes
                    happenings_list = grounded_instance.processes.get_applicable(state)
                    for hp in happenings_list:
                        new_state = new_state.apply_happening(hp, from_state=from_state, create_new_state=new_state is state)

                    # check triggered events again, after applying the effects of events and processes.
                    if constants.DOUBLE_EVENT_CHECK:
                        happenings_list_2 = grounded_instance.events.get_applicable(state)
                        for hp2 in happenings_list_2:
                            new_state = new_state.apply_happening(hp2, from_state=from_state, create_new_state=new_state is state)

                    if new_state is state:
                        new_state = copy.deepcopy(state)
                        new_state.predecessor_hashed = hash(from_state)

                    new_state.time = time_passed
                    new_state.predecessor_action = aa
                else:
                    new_state = state.apply_happening(aa, from_state=from_state)
                self.explored_states += 1

                new_state_hash = hash(VisitedState(new_state))
                if new_state_hash not in self.visited_hashmap and new_state.time <= constants.TIME_HORIZON:
                    if grounded_instance.goals(new_state, constants):
                        self.reached_goal_states.append(new_state)
                        # self.total_visited = len(self.visited_hashmap)
                        if (constants.ANYTIME):
                            time_checkpoint = time.time() - start_solve_time
                            print('[' + str("{:6.2f}".format(time_checkpoint)) + '] ==> found goals: ' + str(len(self.reached_goal_states)))
                        else:
                            return self.reached_goal_states
                    self.visited_hashmap[new_state_hash] = VisitedState(new_state)
                    self.enqueue_state(new_state)

                if self.explored_states % constants.PRINT_INFO == 0:
                    # visi = len(self.visited_hashmap)
                    time_checkpoint = time.time() - start_solve_time
                    print('[' + str("{:6.2f}".format(time_checkpoint)) + '] ==> states explored: ' + str(self.explored_states))
                    print('\t\t\t' + str(round(self.explored_states / time_checkpoint, 2)) + ' states/sec')

                # if constants.PRINT_ALL_STATES:
                #     print(new_state)

            if (time.time() - start_solve_time) >= constants.TIMEOUT:
                if (constants.ANYTIME):
                    return self.reached_goal_states
                return None

        return None

    def enqueue_state(self, n_state):
        self.heuristic.evaluate(n_state)
        self.queue.push(n_state)

        if constants.PRINT_ALL_STATES:
            print(n_state)

    def _get_open_list(self):
        if constants.SEARCH_ASTAR:
            return HeapPriorityList()
        elif constants.SEARCH_DFS:
            return DFSList()
        elif constants.SEARCH_GBFS:
            return HeapPriorityList(Astar=False)
        else:
            # defalut to BFS
            return BFSList()


    def get_trajectory(self, sstate: State):
        plan = []
        curr_v_state = VisitedState(sstate)

        while curr_v_state.state.predecessor_action is not None:
            plan.append((curr_v_state.state.predecessor_action, copy.deepcopy(curr_v_state.state)))
            curr_v_state = self.visited_hashmap[curr_v_state.state.predecessor_hashed]
        plan.reverse()
        return plan
