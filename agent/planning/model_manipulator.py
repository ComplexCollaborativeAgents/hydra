"""
This model manipulates a PDDL+ problem and domain files
"""
from agent.planning.pddl_plus import PddlPlusProblem, PddlPlusDomain
import agent.planning.pddl_plus as pddl_plus
from agent.planning.sb_meta_model import *


class ModelManipulator:
    """ General interface"""

    def __int__(self, pddl_meta_model: MetaModel):
        self.pddl_meta_model = pddl_meta_model

    def apply_change(self):
        raise ValueError("Not implemented yet")


class ManipulateNumericConstant(ModelManipulator):
    """
    Changes the given pddl+ problem file by adding delta to one of the numeric fluent in the initial state.
    Note, this directly modifies the given pddl_problem object, it does not create a clone.
    """

    def __init__(self, pddl_meta_model: MetaModel, fluent_name, delta: float):
        super().__init__(pddl_meta_model)
        self.fluent_to_change = fluent_name
        self.delta = delta

    def apply_change(self):
        old_value = self.pddl_meta_model.constant_numeric_fluents[self.fluent_to_change]
        self.pddl_meta_model.constant_numeric_fluents[self.fluent_to_change] = float(old_value) + self.delta


class ManipulateInitNumericFluent(ManipulateNumericConstant):
    """
    Changes the given pddl+ problem file by adding delta to one of the numeric fluent in the initial state.
    Note, this directly modifies the given pddl_problem object, it does not create a clone.
    """

    def __init__(self, pddl_meta_model: MetaModel, fluent_name, delta: float):
        super().__init__(pddl_meta_model, fluent_name, delta)


class ModifyEffect(ModelManipulator):
    """
    Changes the given pddl+ domain  by adding delta to one of the numeric fluent in the initial state.
    This creates a new PDDL+ domain file and stores its name in the metamodel's domain name field.
    """
    # TODO change interface and behavior entirely (after refactor)

    def __init__(self, action_name, effect_line, new_effect, meta_model: MetaModel):
        super().__init__(self, meta_model)
        self.action_name = action_name
        self.effect_line = effect_line
        self.new_effect = new_effect
        self.old_effect = None

    def apply_change(self):
        # TODO need to figure out how to make changes
        raise NotImplementedError()
        # domain = pddl_meta_model.create_pddl_domain(self.observations[0])
        # action = domain.get_action(self.action_name)
        # if self.effect_line == len(action.effcts) + 1:
        #     # adding a line to the effect
        #     action.effects.append(self.new_effect)
        # elif not self.new_effect:
        #     # assign the value True to a dummy variable == delete effect
        #     action.effects = ['dummy']
        # else:
        #     # replace effect, keep track of old one so we can undo
        #     self.old_effect, action.effects[self.effect_line] = self.new_effect, action.effects[self.effect_line]
        # domain_file_name = domain.name + self.action_name + str(self.effect_line)
        # pddl_meta_model.domain_file_name = domain_file_name
        # PddlDomainExporter().to_file(domain, domain_file_name)
