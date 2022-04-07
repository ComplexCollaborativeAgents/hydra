'''
This model manipulates a PDDL+ problem and domain files
'''
from agent.planning.pddl_plus import PddlPlusProblem, PddlPlusDomain
import agent.planning.pddl_plus as pddl_plus
from agent.planning.sb_meta_model import *

''' General interface'''
class ProblemManipualtor():
    def apply_change(self, pddl_domain, pddl_problem):
        raise ValueError("Not implemented yet")


'''
Changes the given pddl+ problem file by adding delta to one of the numeric fluent in the initial state.
Note, this directly modifies the given pddl_problem object, it does not create a clone.  
'''
class ManipulateInitNumericFluent(ProblemManipualtor):
    def __init__(self, fluent_name, delta: float):
        self.fluent_to_change = fluent_name
        self.delta = delta

    def apply_change(self, pddl_domain, pddl_problem):
        fluent = pddl_plus.get_numeric_fluent(pddl_problem.init, self.fluent_to_change)
        old_value = pddl_plus.get_numeric_fluent_value(fluent)
        fluent[-1]= float(old_value)+self.delta


''' General interface'''
class MetaModelManipualtor():
    def apply_change(self, pddl_meta_model :ScienceBirdsMetaModel):
        raise ValueError("Not implemented yet")


'''
Changes the given pddl+ problem file by adding delta to one of the numeric fluent in the initial state.
Note, this directly modifies the given pddl_problem object, it does not create a clone.  
'''
class ManipulateNumericConstant(MetaModelManipualtor):
    def __init__(self, fluent_name, delta: float):
        self.fluent_to_change = fluent_name
        self.delta = delta

    def apply_change(self, pddl_meta_model :ScienceBirdsMetaModel):
        old_value = pddl_meta_model.constant_numeric_fluents[self.fluent_to_change]
        pddl_meta_model.constant_numeric_fluents[self.fluent_to_change] = float(old_value)+self.delta


class ModifyEffect(MetaModelManipualtor):
    """
    Changes the given pddl+ domain  by adding delta to one of the numeric fluent in the initial state.
    This creates a new PDDL+ domain file and stores its name in the metamodel's domain name field.
    """
    def __init__(self, action_name, effect_line, new_effect, observations):
        self.action_name = action_name
        self.effect_line = effect_line
        self.new_effect = new_effect
        self.old_effect = None
        self.observations = observations

    def apply_change(self, pddl_meta_model :ScienceBirdsMetaModel):
        #TODO need to figure out how to make changes
        # presumably, we create the initial domain from the last observation without novelty?
        domain = pddl_meta_model.create_pddl_domain(self.observations[0])
        action = domain.get_action(self.action_name)
        if self.effect_line == len(action.effcts) + 1:
            # adding a line to the effect
            action.effects.append(self.new_effect)
        elif not self.new_effect:
            # assign the value True to a dummy variable == delete effect
            action.effects = ['dummy']
        else:
            # replace effect, keep track of old one so we can undo
            self.old_effect, action.effects[self.effect_line] = self.new_effect, action.effects[self.effect_line]
        domain_file_name = domain.name + self.action_name + str(self.effect_line)
        pddl_meta_model.domain_file_name = domain_file_name
        PddlDomainExporter().to_file(domain, domain_file_name)




