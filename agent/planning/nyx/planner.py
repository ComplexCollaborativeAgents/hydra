#!/usr/bin/env python
# Four spaces as indentation [no tabs]
import bisect
import collections
import matplotlib.pyplot as plt
from hmac import new

import agent.planning.nyx.heuristic_functions as heuristic_functions
from agent.planning.nyx.PDDL import PDDL_Parser
import settings
import agent.planning.nyx.syntax.constants as constants
import time, copy
from agent.planning.nyx.syntax.visited_state import VisitedState
from agent.planning.nyx.syntax.state import State

# (NOT AVAILABLE YET ON MASTER BRANCH)
import agent.planning.nyx.semantic_attachments.semantic_attachment as semantic_attachment

class Planner:

    #-----------------------------------------------
    # Solve
    #-----------------------------------------------

    # initial_state = None
    # reached_goal_state = None
    # explored_states = 0
    # # total_visited = 0
    # queue = []
    # visited_hashmap = {}

    def __init__(self):
        self.initial_state = None
        self.reached_goal_states = collections.deque(maxlen=constants.TRACKED_PLANS)
        self.explored_states = 0
        # self.total_visited = 0
        self.queue = collections.deque()
        self.visited_hashmap = {}


    def solve(self, domain, problem):

        if constants.PLOT_BIRD_NODE_ORDER:
            node_bird_data = []

        start_solve_time = time.time()
        # Parser
        parser = PDDL_Parser(domain, problem)
        grounded_instance = parser.grounded_instance
        # Parsed data
        state = grounded_instance.init_state
        self.initial_state = grounded_instance.init_state
        heuristic_functions.h_list[constants.CUSTOM_HEURISTIC_ID].notify_initial_state(state)

        print("\t* model parse time: " + str("{:5.4f}".format(time.time() - start_solve_time)) + "s")
        print('\n=================================================\n')
        # Do nothing
        if grounded_instance.goals(state, constants):
            return []

        # Search
        self.visited_hashmap[hash(VisitedState(state))] = VisitedState(state)
        self.queue = collections.deque([state])
        while self.queue:
            state = self.queue.popleft()
            self.explored_states += 1
            if constants.PLOT_BIRD_NODE_ORDER:
                active_bird_string = heuristic_functions.get_active_bird_string(state)
                if active_bird_string is None:
                    node_bird_data.append((98, 19))
                else:
                    node_bird_data.append((state.state_vars["['x_bird'" + active_bird_string],
                                           state.state_vars["['y_bird'" + active_bird_string]))  # Active bird x, y

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
                        new_state = semantic_attachment.external_function(new_state)

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
                        new_state = State(predecessor=state)
                        new_state.predecessor = state
                        new_state.predecessor_hashed = hash(from_state)

                    new_state.time = time_passed
                    new_state.predecessor_action = aa
                else:
                    new_state = state.apply_happening(aa, from_state=from_state)

                visited_state = VisitedState(new_state)
                new_state_hash = hash(visited_state)
                if new_state_hash not in self.visited_hashmap and new_state.time <= constants.TIME_HORIZON and new_state.depth <= constants.DEPTH_LIMIT:
                    if grounded_instance.goals(new_state, constants):
                        # if grounded_instance.problem.metric == ['total-time']:
                        #     new_state.metric = new_state.time
                        # elif grounded_instance.problem.metric == ['total-actions'] or grounded_instance.problem.metric == None:
                        #     new_state.metric = new_state.depth
                        # else:
                        #     new_state.metric = grounded_instance.problem.metric(new_state, constants)
                        self.enqueue_goal(new_state)
                        if not (constants.ANYTIME):
                            if constants.PLOT_BIRD_NODE_ORDER:
                                self.plot_node_expasion(node_bird_data)
                            return self.reached_goal_states

                    if new_state.depth < constants.DEPTH_LIMIT:
                        self.visited_hashmap[new_state_hash] = visited_state
                        self.enqueue_state(new_state)

                if self.explored_states % constants.PRINT_INFO == 0:
                    # visi = len(self.visited_hashmap)
                    time_checkpoint = time.time() - start_solve_time
                    print('\n[' + str("{:6.2f}".format(time_checkpoint)) + '] ==> states explored: ' + str(self.explored_states))
                    print('\t\t ==> exploration rate: ' + str(round(self.explored_states / time_checkpoint, 2)) + ' states/sec')
                    if (constants.ANYTIME):
                        print('\t\t ==> found goals: ' + str(len(self.reached_goal_states)))
                        print('\t\t ==> best metric: {:6.3f}'.format(self.reached_goal_states[0].metric))

            if (time.time() - start_solve_time) >= constants.TIMEOUT:
                if (constants.ANYTIME):
                    if constants.PLOT_BIRD_NODE_ORDER:
                        self.plot_node_expasion(node_bird_data)
                    return self.reached_goal_states
                if constants.PLOT_BIRD_NODE_ORDER:
                    self.plot_node_expasion(node_bird_data)
                return None

        if constants.PLOT_BIRD_NODE_ORDER:
            self.plot_node_expasion(node_bird_data)
        return None



    def enqueue_state(self, n_state):
        if constants.SEARCH_BFS:
            self.queue.append(n_state)
        elif constants.SEARCH_DFS:
            self.queue.appendleft(n_state)
        elif constants.SEARCH_GBFS:
            n_state.h = heuristic_functions.heuristic_function(n_state)
            ''' changing enqueue to bisect.insort ==> needs performance comparison '''
            # self.queue.insert(0, n_state)
            # self.queue = sorted(self.queue, key=lambda elem: (elem.h))

            bisect.insort(self.queue, n_state)

            # self.queue.insert(bisect.bisect_left(self.queue, n_state), n_state)

            # self.queue.appendleft(n_state)
            # self.queue = collections.deque(sorted(self.queue, key=lambda elem: (elem.h)))

        elif constants.SEARCH_ASTAR:
            n_state.h = heuristic_functions.heuristic_function(n_state)
            self.queue.appendleft(n_state)
            self.queue = collections.deque(sorted(self.queue, key=lambda elem: (elem.h + elem.g)))

        if constants.PRINT_ALL_STATES:
            print(n_state)


    def enqueue_goal(self, n_state):
        ''' changing enqueue to bisect.insort ==> needs performance comparison '''
        # print("\n\nNEW STATE METRIC: " + str(n_state.metric))

        if constants.METRIC_MINIMIZE:
            if (len(self.reached_goal_states) < constants.TRACKED_PLANS) or (len(self.reached_goal_states) == constants.TRACKED_PLANS and n_state.metric < self.reached_goal_states[-1].metric):
                self.reached_goal_states.appendleft(n_state)
                self.reached_goal_states = collections.deque(sorted(self.reached_goal_states, key=lambda elem: (elem.metric)), maxlen=constants.TRACKED_PLANS)

        else:
            if (len(self.reached_goal_states) < constants.TRACKED_PLANS) or (len(self.reached_goal_states) == constants.TRACKED_PLANS and n_state.metric > self.reached_goal_states[-1].metric):
                self.reached_goal_states.appendleft(n_state)
                self.reached_goal_states = collections.deque(sorted(self.reached_goal_states, key=lambda elem: (elem.metric), reverse=True), maxlen=constants.TRACKED_PLANS)

        # for st in self.reached_goal_states:
        #     print(st.metric, end=", ")
        # print("\n\n")

    def get_trajectory(self, sstate: State):
        plan = []
        curr_v_state = VisitedState(sstate)

        while curr_v_state.state.predecessor_action is not None:
            plan.insert(0, (curr_v_state.state.predecessor_action, curr_v_state.state))
            curr_v_state = self.visited_hashmap[curr_v_state.state.predecessor_hashed]
        return plan

    def plot_node_expasion(self, bird_xy):
        plt.figure()
        colors, x_data, y_data = [], [], []
        xys = {}
        for c, xy in enumerate(bird_xy):
            colors.append(c)
            x_data.append(xy[0])
            y_data.append(xy[1])
            # if xys.get((xy[0], xy[0])):
            #     xys[(xy[0], xy[0])].append(xy[2])
            # else:
            #     xys[(xy[0], xy[0])] = [xy[2]]
        # for states in xys.values():
        #     if len(states) > 3:
        #         for state in states:
        #             if state.predecessor is not None:
        #                 print(state.copy_counter)
        #                 print(state.time)
        #                 print(state.state_vars)
        #                 print(state.predecessor.time)
        #                 print(state.predecessor.state_vars)
        #                 # print(state.predecessor_action)
        #                 print('*******************')
        print(len(bird_xy))
        print(self.explored_states)
        # print(len(xys))
        plt.figure()
        plt.scatter(x_data, y_data, c=colors)
        plt.title(settings.EXPERIMENT_NAME + ' ' + str(settings.NOVELTY_TYPE))
        plt.show()
