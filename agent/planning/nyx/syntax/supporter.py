import itertools
import re

from agent.planning.nyx.compiler.HappeningMixin import HappeningMixin


class Supporter(HappeningMixin):
    """
    A 'supporter' - created from another happening, set the value of a numeric fluent to an interval based on the PDDL+
    interval relaxation heuristic (see https://ebooks.iospress.nl/publication/44811)
    """
    def __init__(self, name, parameters, preconditions, effects):
        HappeningMixin.__init__(self)
        self.name = name
        self.duration = 0.0
        self.parameters = parameters
        self.preconditions = preconditions
        self.effects = effects

    @property
    def grounded_name(self) -> str:
        res = self.name
        if len(self.parameters) != 0:
            res += ' ' + re.sub(r'[(,\')]', '', str(self.parameters))
        return res

    # -----------------------------------------------
    # to String
    # -----------------------------------------------

    def __str__(self):
        return 'supporter: ' + self.name + \
               '\n  parameters: ' + str(self.parameters) + \
               '\n  preconditions: ' + str([list(i) for i in self.preconditions]) + \
               '\n  effects: ' + str([list(i) for i in self.effects])

    # -----------------------------------------------
    # Equality
    # -----------------------------------------------

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    # -----------------------------------------------
    # Groundify
    # -----------------------------------------------

    def groundify(self, objects, types):
        if not self.parameters:
            yield self
            return
        type_map = []
        variables = []

        for var, type in self.parameters:
            type_stack = [type]
            items = []
            while type_stack:
                t = type_stack.pop()
                if t in objects:
                    items += objects[t]
                elif t in types:
                    type_stack += types[t]
                else:
                    raise Exception('Unrecognized type ' + t)
            type_map.append(items)
            variables.append(var)

        for assignment in itertools.product(*type_map):
            mapping = dict(zip(variables, assignment))
            preconditions = self.copy_replace(self.preconditions, mapping)
            effects = self.copy_replace(self.effects, mapping)
            yield Supporter(self.name, assignment, preconditions, effects)

    # -----------------------------------------------
    # Replace
    # -----------------------------------------------

    def copy_replace(self, element, mapping):
        if isinstance(element, list):
            return [self.copy_replace(e, mapping) for e in element]
        return mapping.get(element, element)

    def __hash__(self):
        return hash(str(self))