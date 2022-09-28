#!/usr/bin/env python
# Four spaces as indentation [no tabs]
import collections
import time

# (NOT AVAILABLE YET ON MASTER BRANCH)
import agent.planning.nyx.semantic_attachments.semantic_attachment as semantic_attachment
import agent.planning.nyx.syntax.constants as constants
from agent.planning.nyx.PDDL import PDDL_Parser
from agent.planning.nyx.heuristic_functions import get_heuristic_function, SBHelpfulAngleHeuristic
from agent.planning.nyx.openlist import BFSList, DFSList, PreferredList, AlternatingList, PriorityList
from agent.planning.nyx.syntax.state import State
from agent.planning.nyx.syntax.visited_state import VisitedState


class Planner:

    # -----------------------------------------------
    # Solve
    # -----------------------------------------------

    def __init__(self):
        self.initial_state = None
        self.reached_goal_states = collections.deque(maxlen=constants.TRACKED_PLANS)
        self.explored_states = 0
        # self.total_visited = 0
        self.visited_hashmap = {}
        self.queue = self._get_open_list()  # TODO get this parameter in some normal way
        self.heuristic = get_heuristic_function(
            constants.CUSTOM_HEURISTIC_ID)  # TODO get this parameter in some normal way

    def solve(self, domain, problem):

        start_solve_time = time.time()
        # Parser
        parser = PDDL_Parser(domain, problem)
        grounded_instance = parser.grounded_instance
        if constants.SB_W_HELPFUL_ACTIONS:  # TODO get this parameter some normal way
            self.heuristic = SBHelpfulAngleHeuristic(blocking_blocks=True)
            pref_list = PreferredList(PriorityList(), self.heuristic)
            self.queue = AlternatingList([pref_list, PriorityList()])
        else:
            self.heuristic = get_heuristic_function(constants.CUSTOM_HEURISTIC_ID,
                                                    groundedPPDL=grounded_instance)  # TODO get this parameter in some normal way
        # Parsed data
        state = grounded_instance.init_state
        self.initial_state = grounded_instance.init_state
        if type(self.heuristic) is list:
            for h in self.heuristic:
                h.notify_initial_state(state)
        else:
            self.heuristic.notify_initial_state(state)

        print("\t* model parse time: " + str("{:5.4f}".format(time.time() - start_solve_time)) + "s")
        print('\n=================================================\n')
        # Do nothing
        if grounded_instance.goals(state, constants):
            return []

        # Search
        self.visited_hashmap[hash(VisitedState(state))] = VisitedState(state)
        self.queue.push(state)
        while self.queue:
            state = self.queue.pop()
            # if state.predecessor_action is not None:  # and state.predecessor_action.name != 'teleport_to':
            #     logging.getLogger("Polycraft").info(f"Expanding state from action: {state.predecessor_action.name}")
            self.explored_states += 1

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
                        new_state = new_state.apply_happening(hp, from_state=from_state,
                                                              create_new_state=new_state is state)

                    # check whether any semantic attachment processes are active, if applicable
                    if constants.SEMANTIC_ATTACHMENT:
                        if new_state is state:
                            new_state = State(t=round(state.time, constants.NUMBER_PRECISION), g=state.g + 1,
                                              predecessor=state, predecessor_action=aa)
                        new_state = semantic_attachment.external_function(new_state)

                    # first check for triggered events, followed by processes
                    happenings_list = grounded_instance.processes.get_applicable(state)
                    for hp in happenings_list:
                        new_state = new_state.apply_happening(hp, from_state=from_state,
                                                              create_new_state=new_state is state)

                    # check triggered events again, after applying the effects of events and processes.
                    if constants.DOUBLE_EVENT_CHECK:
                        happenings_list_2 = grounded_instance.events.get_applicable(state)
                        for hp2 in happenings_list_2:
                            new_state = new_state.apply_happening(hp2, from_state=from_state,
                                                                  create_new_state=new_state is state)

                    if new_state is state:
                        new_state = State(predecessor=state)
                        new_state.predecessor = state
                        new_state.predecessor_hashed = hash(from_state)

                    new_state.time = time_passed
                    new_state.predecessor_action = aa
                else:
                    new_state = state.apply_happening(aa, from_state=from_state)

                happenings_list = grounded_instance.events.get_applicable(new_state)
                for hp in happenings_list:
                    new_state = new_state.apply_happening(hp, from_state=from_state,
                                                          create_new_state=new_state is state)

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
                            return self.reached_goal_states

                    if new_state.depth < constants.DEPTH_LIMIT:
                        self.visited_hashmap[new_state_hash] = visited_state
                        self.enqueue_state(new_state)

                if self.explored_states % constants.PRINT_INFO == 0:
                    # visi = len(self.visited_hashmap)
                    time_checkpoint = time.time() - start_solve_time
                    print('\n[' + str("{:6.2f}".format(time_checkpoint)) + '] ==> states explored: ' + str(
                        self.explored_states))
                    print('\t\t ==> exploration rate: ' + str(
                        round(self.explored_states / time_checkpoint, 2)) + ' states/sec')
                    if constants.ANYTIME:
                        print('\t\t ==> found goals: ' + str(len(self.reached_goal_states)))
                        print('\t\t ==> best metric: {:6.3f}'.format(self.reached_goal_states[0].metric))

            if (time.time() - start_solve_time) >= constants.TIMEOUT:
                if (constants.ANYTIME):
                    return self.reached_goal_states
                return None

        # logger.info(f"Open list exhausted. Found {len(self.reached_goal_states)} plans")
        return None

    def enqueue_state(self, n_state):
        if type(self.heuristic) is list:
            for h in self.heuristic:
                h.evaluate(n_state)
        else:
            self.heuristic.evaluate(n_state)
        self.queue.push(n_state)

        if constants.PRINT_ALL_STATES:
            print(n_state)

    def _get_open_list(self):
        if constants.SEARCH_ASTAR:
            return PriorityList()
        elif constants.SEARCH_DFS:
            return DFSList()
        elif constants.SEARCH_GBFS:
            return PriorityList(Astar=False)
        else:
            # defalut to BFS
            return BFSList()

    def enqueue_goal(self, n_state):
        """ changing enqueue to bisect.insort ==> needs performance comparison """
        # print("\n\nNEW STATE METRIC: " + str(n_state.metric))

        if constants.METRIC_MINIMIZE:
            if (len(self.reached_goal_states) < constants.TRACKED_PLANS) or (
                    len(self.reached_goal_states) == constants.TRACKED_PLANS and n_state.metric <
                    self.reached_goal_states[-1].metric):
                self.reached_goal_states.appendleft(n_state)
                self.reached_goal_states = collections.deque(
                    sorted(self.reached_goal_states, key=lambda elem: (elem.metric)), maxlen=constants.TRACKED_PLANS)

        else:
            if (len(self.reached_goal_states) < constants.TRACKED_PLANS) or (
                    len(self.reached_goal_states) == constants.TRACKED_PLANS and n_state.metric >
                    self.reached_goal_states[-1].metric):
                self.reached_goal_states.appendleft(n_state)
                self.reached_goal_states = collections.deque(
                    sorted(self.reached_goal_states, key=lambda elem: (elem.metric), reverse=True),
                    maxlen=constants.TRACKED_PLANS)

        # for st in self.reached_goal_states:
        #     print(st.metric, end=", ")
        # print("\n\n")

    def get_trajectory(self, v_state: State):
        plan = []
        curr_v_state = VisitedState(v_state)

        while curr_v_state.state.predecessor_action is not None:
            plan.append((curr_v_state.state.predecessor_action, curr_v_state.state))
            curr_v_state = self.visited_hashmap[curr_v_state.state.predecessor_hashed]
        plan.reverse()
        return plan
