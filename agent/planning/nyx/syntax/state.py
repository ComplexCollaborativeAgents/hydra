#!/usr/bin/env python
# Four spaces as indentation [no tabs]

import copy
import math

import agent.planning.nyx.syntax.constants as constants
from agent.planning.nyx.syntax.visited_state import VisitedState


class State:
    # TODO this should be called "node", not "state", because it holds fields like the predecessor and heuristic value.

    # -----------------------------------------------
    # Initialize
    # -----------------------------------------------

    def __init__(self, t=0.0, h=0.0, g=0.0, state_vars=None, predecessor=None, predecessor_action=None):
        if state_vars is None:
            state_vars = {}
        if predecessor is not None:
            if isinstance(predecessor, VisitedState):
                self.predecessor_hashed = hash(predecessor)
                predecessor = predecessor.state
            else:
                self.predecessor_hashed = hash(VisitedState(predecessor))
            self._time = round(predecessor.time,constants.NUMBER_PRECISION)
            self._h = h
            self._g = (predecessor.g+1) if constants.TRACK_G else 0.0
            self.metric = math.inf
            self.depth = predecessor.depth + 1
            self.state_vars = copy.copy(predecessor.state_vars)
            self.predecessor_action = predecessor_action
        else:
            self._time = round(t,constants.NUMBER_PRECISION)
            self._h = h
            self._g = g
            self.metric = math.inf
            self.depth = 0
            self.state_vars = state_vars
            self.predecessor_hashed = None
            self.predecessor_action = None

    # -----------------------------------------------
    # to String
    # -----------------------------------------------

    def __str__(self):
        return '\tstate: ' + \
               '\n\t  time: ' + str(self.time) + \
               '\n\t  h: ' + str(self.h) + \
               '\n\t  g: ' + str(self.g) + \
               '\n\t  depth: ' + str(self.depth) + \
               '\n\t  state vars: ' + str([list(i) for i in self.state_vars.items()])

    # -----------------------------------------------
    # Equality
    # -----------------------------------------------

    def __eq__(self, other):
        return other and (self.h+self.g) == (other.h+other.g)

    def __lt__(self, other):
        return other and (self.h+self.g) < (other.h+other.g)

    def __ne__(self, other):
        return not self.__eq__(other)

    # -----------------------------------------------
    # Hashing
    # -----------------------------------------------

    def __hash__(self):
        return hash(self.h+self.g)

    # -----------------------------------------------
    # Heuristic Estimate
    # -----------------------------------------------

    @property
    def h(self):
        return self._h

    @h.setter
    def h(self, value):
        """
        Update heuristic value, but only if it dominates the one we already have.
        """
        self._h = value

    # -----------------------------------------------
    # Traversed Depth
    # -----------------------------------------------
    @property
    def g(self):
        return self._g

    @g.setter
    def g(self, val):
        self._g = val

    # -----------------------------------------------
    # time
    # -----------------------------------------------

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, val):
        self._time = val

    # -----------------------------------------------
    # instantiation of state variables from problem init
    # -----------------------------------------------

    def instantiate(self, init_list):
        for iv in init_list:
            if iv[0] == '=':
                # print(str(iv[1]) + ' is a function with value = ' + iv[-1])
                self.state_vars[str(iv[1])] = round(float(iv[-1]),constants.NUMBER_PRECISION)
            elif iv[0] == 'not':
                continue
            else:
                # print(str(iv) + ' is a true predicate')
                self.state_vars[str(iv)] = True

    # -----------------------------------------------
    # Get a list of all applicable actions for this state
    # -----------------------------------------------

    def get_applicable_happenings(self, grounded_happening_list):

        applicables = []

        for act in grounded_happening_list:
            if act.preconditions_func(self, constants):
            # if self.evaluate_preconditions(act):
                # print('\nAPPLICABLE ACTION: ' + act.name)
                # print(act)
                applicables.append(act)
            # else:
                # print('\nNOT APPLICABLE: ' + act.name)
                # print(act)

        return applicables

    # -----------------------------------------------
    # Check if preconditions of an action are satisfied in this state
    # (currently conjunctive preconditions only)
    # -----------------------------------------------

    def evaluate_preconditions(self, happening):
        for prec in happening.preconditions:
            if self.eval_stmt(prec) == False:
                # print('FALSE PRECONDITION: ' + str(prec) + ' in ' + str(action.name) + '[' + str(action.parameters) + ']')
                return False
        return True

    def eval_stmt(self, precondition):

        prec_tokens = copy.deepcopy(precondition)

        # print('\n')
        # print(prec_tokens)

        if str(prec_tokens) in self.state_vars:
            # print('\n\nFOUND A PREDICATE/FUNCTION in STATE VARIABLESSSSSSSSSS')
            # print(str(prec_tokens) + ' = ' + str(self.state_vars[str(prec_tokens)]))
            if type(self.state_vars[str(prec_tokens)]) == bool:
                return self.state_vars[str(prec_tokens)]
            else:
                return round(self.state_vars[str(prec_tokens)],constants.NUMBER_PRECISION)

        if isinstance(prec_tokens, list):
            token = prec_tokens.pop(0)
        else:
            token = prec_tokens

        if str(token) in self.state_vars:
            # print('\n\nFOUND A PREDICATE/FUNCTION in STATE VARIABLESSSSSSSSSS')
            # print(str(token) + ' = ' + str(self.state_vars[str(token)]))
            if type(self.state_vars[str(token)]) == bool:
                return self.state_vars[str(token)]
            else:
                return round(self.state_vars[str(token)],constants.NUMBER_PRECISION)
        if token == 'or':
            for e in prec_tokens:
                if self.eval_stmt(e) == True:
                    return True
            return False
        elif token == '+':
            return round(self.eval_stmt(prec_tokens.pop(0)), constants.NUMBER_PRECISION) + round(self.eval_stmt(prec_tokens.pop(0)), constants.NUMBER_PRECISION)
        elif token == '-':
            return round(self.eval_stmt(prec_tokens.pop(0)), constants.NUMBER_PRECISION) - round(self.eval_stmt(prec_tokens.pop(0)), constants.NUMBER_PRECISION)
        elif token == '*':
            return round(self.eval_stmt(prec_tokens.pop(0)), constants.NUMBER_PRECISION) * round(self.eval_stmt(prec_tokens.pop(0)), constants.NUMBER_PRECISION)
        elif token == '/':
            return round(self.eval_stmt(prec_tokens.pop(0)), constants.NUMBER_PRECISION) / round(self.eval_stmt(prec_tokens.pop(0)), constants.NUMBER_PRECISION)
        elif token == '=':
            return round(self.eval_stmt(prec_tokens.pop(0)), constants.NUMBER_PRECISION) == round(self.eval_stmt(prec_tokens.pop(0)), constants.NUMBER_PRECISION)
        elif token == '>=':
            return round(self.eval_stmt(prec_tokens.pop(0)), constants.NUMBER_PRECISION) >= round(self.eval_stmt(prec_tokens.pop(0)), constants.NUMBER_PRECISION)
        elif token == '<=':
            return round(self.eval_stmt(prec_tokens.pop(0)), constants.NUMBER_PRECISION) <= round(self.eval_stmt(prec_tokens.pop(0)), constants.NUMBER_PRECISION)
        elif token == '>':
            return round(self.eval_stmt(prec_tokens.pop(0)), constants.NUMBER_PRECISION) > round(self.eval_stmt(prec_tokens.pop(0)), constants.NUMBER_PRECISION)
        elif token == '<':
            return round(self.eval_stmt(prec_tokens.pop(0)), constants.NUMBER_PRECISION) < round(self.eval_stmt(prec_tokens.pop(0)), constants.NUMBER_PRECISION)
        elif token == 'not':
            # print(prec_tokens)
            return not self.eval_stmt(prec_tokens)
        elif isinstance(token, list):
            return self.eval_effect(token)
        else:
            # must be just a number
            return round(float(token),constants.NUMBER_PRECISION)

    # -----------------------------------------------
    # Apply a grounded action/happening and return the resulting successor state
    # -----------------------------------------------

    def apply_happening(self, happening, from_state=None, create_new_state=True):
        # if happening.name == 'time-passing':
        #     successor = State(t=self.time+constants.DELTA_T, g=self.g+1, predecessor=self, predecessor_action=happening)
        # else:
        if create_new_state:
            predecessor = self if from_state is None else from_state
            successor = State(t=self.time, g=self.g + 1, predecessor=predecessor, predecessor_action=happening)
        else:
            successor = self
        happening.effects_func(successor, constants)

        # for eff in happening.effects:
            # successor.eval_effect(eff)
            # print("\nSELF STATE:")
            # print(self)
        return successor

    def eval_effect(self, effect):

        eff_tokens = copy.deepcopy(effect)

        # print('\n')
        # print(eff_tokens)

        if str(eff_tokens) in self.state_vars:
            if isinstance(self.state_vars[str(eff_tokens)], bool):
                self.state_vars[str(eff_tokens)] = True
                return
            else:
                return round(float(self.state_vars[str(eff_tokens)]),constants.NUMBER_PRECISION)
            return

        if eff_tokens == '#t' or str(eff_tokens) == '#t':
            return round(constants.DELTA_T,constants.NUMBER_PRECISION)

        if isinstance(eff_tokens, list):
            token = eff_tokens.pop(0)
        else:
            token = eff_tokens

        if str(token) in self.state_vars:
            if isinstance(self.state_vars[str(token)], bool):
                self.state_vars[str(token)] = True
                return
            else:
                return round(float(self.state_vars[str(token)]),constants.NUMBER_PRECISION)


        if token == '+':
            return round(self.eval_effect(eff_tokens[0]) + self.eval_effect(eff_tokens[1]), constants.NUMBER_PRECISION)
        elif token == '-':
            return round(self.eval_effect(eff_tokens[0]) - self.eval_effect(eff_tokens[1]), constants.NUMBER_PRECISION)
        elif token == '*':
            return round(self.eval_effect(eff_tokens[0]) * self.eval_effect(eff_tokens[1]), constants.NUMBER_PRECISION)
        elif token == '/':
            return round(self.eval_effect(eff_tokens[0]) / self.eval_effect(eff_tokens[1]), constants.NUMBER_PRECISION)
        elif token == 'assign':
            self.state_vars[str(eff_tokens[0])] = round(self.eval_effect(eff_tokens[1]), constants.NUMBER_PRECISION)
        elif token == 'increase':
            self.state_vars[str(eff_tokens[0])] += round(self.eval_effect(eff_tokens[1]), constants.NUMBER_PRECISION)
            self.state_vars[str(eff_tokens[0])] = round(self.eval_effect(eff_tokens[0]), constants.NUMBER_PRECISION)
        elif token == 'decrease':
            self.state_vars[str(eff_tokens[0])] -= round(self.eval_effect(eff_tokens[1]), constants.NUMBER_PRECISION)
        elif token == 'scale-up':
            self.state_vars[str(eff_tokens[0])] *= round(self.eval_effect(eff_tokens[1]), constants.NUMBER_PRECISION)
        elif token == 'scale-down':
            self.state_vars[str(eff_tokens[0])] /= round(self.eval_effect(eff_tokens[1]), constants.NUMBER_PRECISION)
        elif token == 'not':
            self.state_vars[str(eff_tokens[0])] = False
        elif isinstance(token, list):
            return self.eval_effect(token)
        elif token == '#t':
            return round(constants.DELTA_T,constants.NUMBER_PRECISION)
        else:
            # must be just a number
            return round(float(token),constants.NUMBER_PRECISION)

    # -----------------------------------------------
    # Check all goal conditions
    # -----------------------------------------------

    def is_goal(self, goal_conditions):
        # print(str(self.state_vars["['elapsed_time']"]) + ' == ' + str(self.time))
        for gc in goal_conditions:
            if self.eval_stmt(gc) == False:
                return False
        return True
