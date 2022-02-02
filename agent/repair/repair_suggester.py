from typing import List

from agent.planning.meta_model import MetaModel
from agent.planning.nyx.syntax.action import Action


class RepairSuggestor:
    """
    Suggests repairs to a pddl+ model based on differences between a planned trace and observations.
    """
    def __init__(self, meta_model:MetaModel, domain, problem):
        self.domain = domain
        self.problem = problem  # do I need these variables? who knows
        self.meta_model = meta_model

    def suggest_repair(self, pre_state, plan_state, obs_state, actions: List[Action], delta_t):
        possible_repairs = []
        for action in actions:
            # parse the action somehow
            for cond in action.preconditions:
                for const in self.meta_model.repairable_constants:
                    # Is the constant in the condition?
                    # What value of the constant would flip the condition?
            for effect in action.effects:
                for const in self.meta_model.repairable_constants:
                    # Is the constant in the expression?
                    # what value of the constant would make the plan state match the observed state?
                    #   what variable(s) are affected?
                    #   what is the value difference?
                    #   is there any reasonable way to work backwards from the value difference to the constant?



        return possible_repairs


