import itertools

import numpy as np
from interval import inf

import agent.planning.nyx.syntax.constants as constants
from agent.planning.nyx.PDDL import GroundedPDDLInstance
from agent.planning.nyx.abstract_heuristic import AbstractHeuristic
from agent.planning.nyx.syntax.state import State
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
        def __init__(self, **kwargs):
            State.__init__(self, **kwargs)
            if kwargs.get('predecessor') and not isinstance(kwargs['predecessor'], IntervalHeuristic.IntervalState):
                # Convert to intervals. This only happens in one place, maybe move it there?
                for key, val in self.state_vars.items():
                    if type(val) is not bool:
                        self.state_vars[key] = ComparableInterval(val)
                self.state_vars['pos_inf'] = ComparableInterval[1, inf]
                self.state_vars['neg_inf'] = ComparableInterval[-inf, 1]

        def apply_happening(self, happening, from_state=None, create_new_state=True):
            if create_new_state:
                predecessor = self if from_state is None else from_state
                successor = IntervalHeuristic.IntervalState(predecessor=predecessor, predecessor_action=happening)
            else:
                successor = self
            happening.effects_func(successor, constants)
            return successor

    def __init__(self, g_pddl: GroundedPDDLInstance):
        """
        Reads the domain file and creates supporters for all actions.
        """
        self.cached_vales = dict()
        self.grounded_domain = g_pddl

        self.supporters = set()

    def notify_initial_state(self, node):
        for happening in itertools.chain(self.grounded_domain.actions, self.grounded_domain.events,
                                         self.grounded_domain.processes):
            for effect in happening.effects:
                first_token = effect[0] if isinstance(effect, list) else effect
                if first_token in IntervalHeuristic.NUMERIC_OPERATORS:
                    self._create_supporters(happening.parameters, happening.preconditions, effect)
                else:
                    # effect is on predicate
                    self._relaxed_effect(happening.name + '_relaxed', happening.parameters,
                                         happening.preconditions, effect)

        # TODO build new precondition tree?

    def _create_supporters(self, parameters, conditions, og_effect):
        """
        Creates supporters for a happening.
        """
        effect = og_effect.copy()
        if effect[0] == 'assign':
            # transform assignments to additive effects
            effect[0] = 'increase'
            effect[2] = ['-', effect[2], effect[1]]
        if effect[0] == 'increase':
            #  multiply by special state variables with values of [-inf, 1] and [1, inf] to get desired intervals.
            self.supporters.add(
                Supporter('increase_inf', parameters, conditions + [['>', effect[2], '0']],
                          [['assign', effect[1], ['*', effect[1], 'pos_inf']]]))
            self.supporters.add(
                Supporter('increase_neg_inf', parameters, conditions + [['<', effect[2], '0']],
                          [['assign', effect[1], ['*', effect[1], 'neg_inf']]]))
        elif effect[0] == 'decrease':
            self.supporters.add(
                Supporter('decrease_inf', parameters, conditions + [['>', effect[2], '0']],
                          [['assign', effect[1], ['*', effect[1], 'neg_inf']]]))
            self.supporters.add(
                Supporter('decrease_neg_inf', parameters, conditions + [['<', effect[2], '0']],
                          [['assign', effect[1], ['*', effect[1], 'pos_inf']]]))

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
        state = IntervalHeuristic.IntervalState(predecessor=node)
        state_updated = True
        happenings_applied = 0
        while supporters and state_updated and not self.grounded_domain.goals(state, constants):
            state_updated = False
            for happening in state.get_applicable_happenings(supporters):
                # TODO This is the first place to look to improve performance (try to use Alex's precondition trees)
                state_updated = True
                happenings_applied += 1
                state.apply_happening(happening)
                supporters.remove(happening)

        if self.grounded_domain.goals(state, constants):
            value = happenings_applied
        else:
            value = np.inf
        node.h = value
        return node.h
