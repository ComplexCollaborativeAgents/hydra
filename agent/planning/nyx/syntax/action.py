#!/usr/bin/env python
# Four spaces as indentation [no tabs]

import itertools
import re

from agent.planning.nyx.compiler.HappeningMixin import HappeningMixin


class Action(HappeningMixin):

    #-----------------------------------------------
    # Initialize
    #-----------------------------------------------

    def __init__(self, name, parameters, preconditions, effects):
        HappeningMixin.__init__(self)

        def frozenset_of_tuples(data):
            print("ACT-data: " + str(data))
            return frozenset([tuple(t) for t in data])
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

    #-----------------------------------------------
    # to String
    #-----------------------------------------------

    def __str__(self):
        return 'action: ' + self.name + \
        '\n  parameters: ' + str(self.parameters) + \
        '\n  preconditions: ' + str([list(i) for i in self.preconditions]) + \
        '\n  effects: ' + str([list(i) for i in self.effects])

    __repr__ = __str__

    #-----------------------------------------------
    # Equality
    #-----------------------------------------------

    def __eq__(self, other): 
        return self.__dict__ == other.__dict__

    #-----------------------------------------------
    # Groundify
    #-----------------------------------------------

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
            yield Action(self.name, assignment, preconditions, effects)

    #-----------------------------------------------
    # Replace
    #-----------------------------------------------

    def copy_replace(self, element, mapping):
        if isinstance(element, list):
            return [self.copy_replace(e, mapping) for e in element]
        return mapping.get(element, element)
