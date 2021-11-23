import copy
import numbers
import re

import numpy as np
import itertools

from interval import inf

from agent.planning.nyx.compiler.HappeningMixin import HappeningMixin

from agent.planning.nyx.PDDL import GroundedPDDLInstance
from agent.planning.nyx.abstract_heuristic import AbstractHeuristic
from agent.planning.nyx.syntax.state import State
import agent.planning.nyx.syntax.constants as constants
from agent.planning.nyx.syntax.supporter import Supporter
from utils.comparable_interval import ComparableInterval


class IntervalHeuristic(AbstractHeuristic):
    """
    Implements the interval-based relaxation heuristic from https://ebooks.iospress.nl/publication/44811.
    """

    NUMERIC_OPERATORS = {'assign', 'increase', 'decrease', 'scale-up', 'scale-down'}

    class IntervalState(State):
        """
        A state that stores intervals for each variable, for the heuristic's ARPG.
        """
        def __init__(self, state):
            State.__init__(self, t=state.time, h=state.h, g=state.g, state_vars=state.state_vars)
            for key, val in self.state_vars.items():
                if type(val) is not bool:
                    self.state_vars[key] = ComparableInterval(val) # In theory, all the regular math works on these.
            self.state_vars['pos_inf'] = ComparableInterval[1, inf]
            self.state_vars['neg_inf'] = ComparableInterval[-inf, 1]

        # TODO should sub-class happening named Supporters to return interval? will need entirely new JIT, because the
        #   current one compiles all the effects into a single function body, exactly the way you'd expect for efficiency.
        #   Should I have a JIT at all? start without and add later if we use this?
        #   New precondition tree?
        #  Also need to remove predicate delete effects: not set things to False

    def __init__(self, g_pddl: GroundedPDDLInstance):
        """
        Reads the domain file and creates supporters for all actions.
        """
        self.cached_vales = dict()
        self.grounded_domain = g_pddl

        self.supporters = set()

    def notify_initial_state(self, node):
        #  TODO Need to figure out if an effect is numeric

        for happening in itertools.chain(self.grounded_domain.actions, self.grounded_domain.events,
                                         self.grounded_domain.processes):
            for effect in happening.effects:
                first_token = effect[0] if isinstance(effect, list) else effect
                if first_token in IntervalHeuristic.NUMERIC_OPERATORS:
                    self._create_supporters(happening.parameters, happening.preconditions, effect)
                else:
                    # effect is on predicate
                    self._relaxed_effect(happening.name + '_relaxed', happening.parameters, happening.preconditions, effect)

        # TODO build new precondition tree?

    def _create_supporters(self, parameters, conditions, og_effect):
        """
        Creates supporters for a happening.
        """
        effect = og_effect.copy()
        if effect[0] == 'assign':
            # transform assignments to additive effects
            effect[0] = 'increase'
            effect[2] = ['-', effect[2], [effect[1]]]
        if effect[0] == 'increase':
            #  multiply by special state variables with values of [-inf, 1] and [1, inf] to get desired intervals.
            self.supporters.add(
                Supporter('increase_inf', parameters, conditions + [['>', effect[2], '0']], ['*', effect[1], 'pos_inf']))
            self.supporters.add(
                Supporter('increase_neg_inf', parameters, conditions + [['<', effect[2], '0']], ['*', effect[1], 'neg_inf']))
        elif effect[0] == 'decrease':
            self.supporters.add(
                Supporter('decrease_inf', parameters, conditions + [['>', effect[2], '0']], ['*', effect[1], 'neg_inf']))
            self.supporters.add(
                Supporter('decrease_neg_inf', parameters, conditions + [['<', effect[2], '0']], ['*', effect[1], 'pos_inf']))

    def _relaxed_effect(self, name, parameters, conditions, effect):
        """
        Relaxed delete effects for predicates (=removes delete effects)
        """
        if isinstance(effect, list) and effect[0] == 'not':
            return
        self.supporters.add(Supporter(name, parameters, conditions, effect))

    def notify_expanded(self, node):
        """
        Returns False if node is a dead-end, so it doesn't have to be expanded.
        """
        # Nodes can only be expanded when coming out of the fringe, so must have been evaluated already.
        return not np.isinf(self.cached_vales[node])

    def evaluate(self, node: State):
        """
        Estimates reachability and computes heuristic simultaneously.
        If goal is reachable, the heuristic is how many actions were applied to reach it (not admissible or
         particularly accurate; see paper). Otherwise, return infinity.
        """
        supporters = self.supporters.copy()
        state = IntervalHeuristic.IntervalState(node)
        state_updated = True
        happenings_applied = 0
        while supporters and state_updated and not self.grounded_domain.goals(state, constants):
            state_updated = False
            for happening in state.get_applicable_happenings(supporters):  # TODO This line is gonna kill our performance
                state_updated = True
                happenings_applied +=1
                state.apply_happening(happening)
                supporters.remove(happening)

        if self.grounded_domain.goals(state, constants):
            value = happenings_applied
        else:
            value = np.inf
        node.h = value
        return node.h