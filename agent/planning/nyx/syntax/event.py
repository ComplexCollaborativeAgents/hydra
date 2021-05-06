#!/usr/bin/env python
# Four spaces as indentation [no tabs]

import itertools
import copy

class Event:

    #-----------------------------------------------
    # Initialize
    #-----------------------------------------------

    def __init__(self, name, parameters, preconditions, effects, extensions = None):
        def frozenset_of_tuples(data):
            print("EV-data: " + str(data))
            return frozenset([tuple(t) for t in data])
        self.name = name
        self.parameters = parameters
        self.preconditions = preconditions
        self.effects = effects

    #-----------------------------------------------
    # to String
    #-----------------------------------------------

    def __str__(self):
        return 'event: ' + self.name + \
        '\n  parameters: ' + str(self.parameters) + \
        '\n  preconditions: ' + str([list(i) for i in self.preconditions]) + \
        '\n  effects: ' + str([list(i) for i in self.effects])

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
            preconditions = self.replace(copy.deepcopy(self.preconditions), variables, assignment)
            effects = self.replace(copy.deepcopy(self.effects), variables, assignment)
            yield Event(self.name, assignment, preconditions, effects)

    #-----------------------------------------------
    # Replace
    #-----------------------------------------------

    def replace(self, group, vars, asses):
        g = []
        var_ass_map = dict(zip(list(vars), asses))
        # print("map: " + str(var_ass_map))
        for pred in group.copy():
            # print('pred: ' + str(pred))
            list_pred = list(pred)
            for v in vars:
                self.nestrepl(list_pred, v, var_ass_map[v])
            g.append(list_pred)
        return g

    def nestrepl(self, lst, what, repl):

        # print('predicate: ' + str(lst) + ' replacing ' + str(what) + ' with ' + str(repl))
        for index, item in enumerate(lst):
            if type(item) == list:
                self.nestrepl(item, what, repl)
            else:
                if item == what:
                    lst[index] = repl