#!/usr/bin/env python
# Four spaces as indentation [no tabs]

import re
from agent.planning.nyx.syntax.action import Action
from agent.planning.nyx.syntax.event import Event
from agent.planning.nyx.syntax.process import Process
from agent.planning.nyx.syntax.state import State
import agent.planning.nyx.syntax.constants as constants
import itertools
import copy


class PDDL_Parser:

    SUPPORTED_REQUIREMENTS = [':strips', ':adl', ':negative-preconditions', ':typing', ':time', ':fluents', ':continuous-effects']
    init_state = None
    grounded_actions = []
    grounded_events = []
    grounded_processes = []

    def __init__(self, domain_file, problem_file):
        self.scan_tokens(domain_file)
        self.scan_tokens(problem_file)
        self.parse_domain(domain_file)
        self.parse_problem(problem_file)
        self.set_grounded_actions()
        self.set_grounded_processes()
        self.set_grounded_events()
        self.set_init_state()



    #-----------------------------------------------
    # Tokens
    #-----------------------------------------------

    def scan_tokens(self, filename):
        with open(filename,'r') as f:
            # Remove single line comments
            str = re.sub(r';.*$', '', f.read(), flags=re.MULTILINE).lower()
        # Tokenize
        stack = []
        list = []
        for t in re.findall(r'[()]|[^\s()]+', str):
            if t == '(':
                stack.append(list)
                list = []
            elif t == ')':
                if stack:
                    l = list
                    list = stack.pop()
                    list.append(l)
                else:
                    raise Exception('Missing open parentheses')
            else:
                list.append(t)
        if stack:
            raise Exception('Missing close parentheses')
        if len(list) != 1:
            raise Exception('Malformed expression')
        return list[0]

    #-----------------------------------------------
    # Parse domain
    #-----------------------------------------------

    def parse_domain(self, domain_filename):
        tokens = self.scan_tokens(domain_filename)
        if type(tokens) is list and tokens.pop(0) == 'define':
            self.domain_name = 'unknown'
            self.requirements = []
            self.types = {}
            self.objects = {}
            self.actions = []
            self.events = []
            self.processes = []
            self.predicates = {}
            self.functions = {}
            while tokens:
                group = tokens.pop(0)
                t = group.pop(0)
                if t == 'domain':
                    self.domain_name = group[0]
                elif t == ':requirements':
                    for req in group:
                        if req == ':time':
                            constants.TEMPORAL_DOMAIN = True
                        if not req in self.SUPPORTED_REQUIREMENTS:
                            raise Exception('Requirement ' + req + ' not supported')
                    self.requirements = group
                elif t == ':constants':
                    self.parse_objects(group, t)
                elif t == ':predicates':
                    self.parse_predicates(group)
                elif t == ':functions':
                    self.parse_functions(group)
                elif t == ':types':
                    self.parse_types(group)
                elif t == ':action':
                    self.parse_action(group)
                elif t == ':event':
                    self.parse_event(group)
                elif t == ':process':
                    self.parse_process(group)
                else: self.parse_domain_extended(t, group)
        else:
            raise Exception('File ' + domain_filename + ' does not match domain pattern')

    def parse_domain_extended(self, t, group):
        print(str(t) + ' is not recognized in domain')

    #-----------------------------------------------
    # Parse hierarchy
    #-----------------------------------------------

    def parse_hierarchy(self, group, structure, name, redefine):
        list = []
        while group:
            if redefine and group[0] in structure:
                raise Exception('Redefined supertype of ' + group[0])
            elif group[0] == '-':
                if not list:
                    raise Exception('Unexpected hyphen in ' + name)
                group.pop(0)
                type = group.pop(0)
                if not type in structure:
                    structure[type] = []
                structure[type] += list
                list = []
            else:
                list.append(group.pop(0))
        if list:
            if not 'object' in structure:
                structure['object'] = []
            structure['object'] += list

    #-----------------------------------------------
    # Parse objects
    #-----------------------------------------------

    def parse_objects(self, group, name):
        self.parse_hierarchy(group, self.objects, name, False)

    # -----------------------------------------------
    # Parse types
    # -----------------------------------------------

    def parse_types(self, group):
        self.parse_hierarchy(group, self.types, 'types', True)

    #-----------------------------------------------
    # Parse predicates
    #-----------------------------------------------

    def parse_predicates(self, group):
        for pred in group:
            predicate_name = pred.pop(0)
            if predicate_name in self.predicates:
                raise Exception('Predicate ' + predicate_name + ' redefined')
            arguments = {}
            untyped_variables = []
            while pred:
                t = pred.pop(0)
                if t == '-':
                    if not untyped_variables:
                        raise Exception('Unexpected hyphen in predicates')
                    type = pred.pop(0)
                    while untyped_variables:
                        arguments[untyped_variables.pop(0)] = type
                else:
                    untyped_variables.append(t)
            while untyped_variables:
                arguments[untyped_variables.pop(0)] = 'object'
            self.predicates[predicate_name] = arguments

    # -----------------------------------------------
    # Parse functions
    # -----------------------------------------------

    def parse_functions(self, group):
        for func in group:
            function_name = func.pop(0)
            if function_name in self.predicates:
                raise Exception('Function ' + function_name + ' redefined')
            arguments = {}
            untyped_variables = []
            while func:
                t = func.pop(0)
                if t == '-':
                    if not untyped_variables:
                        raise Exception('Unexpected hyphen in functions')
                    type = func.pop(0)
                    while untyped_variables:
                        arguments[untyped_variables.pop(0)] = type
                else:
                    untyped_variables.append(t)
            while untyped_variables:
                arguments[untyped_variables.pop(0)] = 'object'
            self.functions[function_name] = arguments

    #-----------------------------------------------
    # Parse action
    #-----------------------------------------------

    def parse_action(self, group):
        name = group.pop(0)
        if not type(name) is str:
            raise Exception('Action without name definition')
        for act in self.actions:
            if act.name == name:
                raise Exception('Action ' + name + ' redefined')
        parameters = []
        preconditions = []
        effects = []
        extensions = None
        while group:
            t = group.pop(0)
            if t == ':parameters':
                if not type(group) is list:
                    raise Exception('Error with ' + name + ' parameters')
                parameters = []
                untyped_parameters = []
                p = group.pop(0)
                while p:
                    t = p.pop(0)
                    if t == '-':
                        if not untyped_parameters:
                            raise Exception('Unexpected hyphen in ' + name + ' parameters')
                        ptype = p.pop(0)
                        while untyped_parameters:
                            parameters.append([untyped_parameters.pop(0), ptype])
                    else:
                        untyped_parameters.append(t)
                while untyped_parameters:
                    parameters.append([untyped_parameters.pop(0), 'object'])
            elif t == ':precondition':
                # preconditions = group.pop(0)
                self.split_predicates(group.pop(0), preconditions, name, ' preconditions')
            elif t == ':effect':
                # effects = group.pop(0)
                self.split_predicates(group.pop(0), effects, name, ' effects')
            else: extensions = self.parse_action_extended(t, group)
        self.actions.append(Action(name, parameters, preconditions, effects))

    def parse_action_extended(self, t, group):
        print(str(t) + ' is not recognized in action')

    # -----------------------------------------------
    # Parse event
    # -----------------------------------------------

    def parse_event(self, group):
        name = group.pop(0)
        if not type(name) is str:
            raise Exception('Event without name definition')
        for eve in self.events:
            if eve.name == name:
                raise Exception('Event ' + name + ' redefined')
        parameters = []
        preconditions = []
        effects = []
        extensions = None
        while group:
            t = group.pop(0)
            if t == ':parameters':
                if not type(group) is list:
                    raise Exception('Error with ' + name + ' parameters')
                parameters = []
                untyped_parameters = []
                p = group.pop(0)
                while p:
                    t = p.pop(0)
                    if t == '-':
                        if not untyped_parameters:
                            raise Exception('Unexpected hyphen in ' + name + ' parameters')
                        ptype = p.pop(0)
                        while untyped_parameters:
                            parameters.append([untyped_parameters.pop(0), ptype])
                    else:
                        untyped_parameters.append(t)
                while untyped_parameters:
                    parameters.append([untyped_parameters.pop(0), 'object'])
            elif t == ':precondition':
                self.split_predicates(group.pop(0), preconditions, name, ' preconditions')
            elif t == ':effect':
                self.split_predicates(group.pop(0), effects, name, ' effects')
            else:
                extensions = self.parse_event_extended(t, group)
        self.events.append(Event(name, parameters, preconditions, effects, extensions))

    def parse_event_extended(self, t, group):
        print(str(t) + ' is not recognized in event')

    # -----------------------------------------------
    # Parse process
    # -----------------------------------------------

    def parse_process(self, group):
        name = group.pop(0)
        if not type(name) is str:
            raise Exception('Process without name definition')
        for pro in self.processes:
            if pro.name == name:
                raise Exception('Process ' + name + ' redefined')
        parameters = []
        preconditions = []
        effects = []
        extensions = None
        while group:
            t = group.pop(0)
            if t == ':parameters':
                if not type(group) is list:
                    raise Exception('Error with ' + name + ' parameters')
                parameters = []
                untyped_parameters = []
                p = group.pop(0)
                while p:
                    t = p.pop(0)
                    if t == '-':
                        if not untyped_parameters:
                            raise Exception('Unexpected hyphen in ' + name + ' parameters')
                        ptype = p.pop(0)
                        while untyped_parameters:
                            parameters.append([untyped_parameters.pop(0), ptype])
                    else:
                        untyped_parameters.append(t)
                while untyped_parameters:
                    parameters.append([untyped_parameters.pop(0), 'object'])
            elif t == ':precondition':
                self.split_predicates(group.pop(0), preconditions, name, ' preconditions')
            elif t == ':effect':
                self.split_predicates(group.pop(0), effects, name, ' effects')
            else:
                extensions = self.parse_process_extended(t, group)
        self.processes.append(Process(name, parameters, preconditions, effects, extensions))

    def parse_process_extended(self, t, group):
        print(str(t) + ' is not recognized in process')

    #-----------------------------------------------
    # Parse problem
    #-----------------------------------------------

    def parse_problem(self, problem_filename):
        def frozenset_of_tuples(data):
            return frozenset([tuple(t) for t in data])
        tokens = self.scan_tokens(problem_filename)
        if type(tokens) is list and tokens.pop(0) == 'define':
            self.problem_name = 'unknown'
            self.initialized_problem_state_variables = frozenset()
            self.goals = frozenset()
            self.metric = 'unknown'
            while tokens:
                group = tokens.pop(0)
                t = group.pop(0)
                if t == 'problem':
                    self.problem_name = group[0]
                elif t == ':domain':
                    if self.domain_name != group[0]:
                        raise Exception('Different domain specified in problem file')
                elif t == ':requirements':
                    pass # Ignore requirements in problem, parse them in the domain
                elif t == ':objects':
                    self.parse_objects(group, t)
                elif t == ':init':
                    # print('\nREAD INIT STATE\n')
                    # pprint.pprint(group)
                    self.initialized_problem_state_variables = group
                elif t == ':goal':
                    goals = []
                    self.split_predicates(group[0], goals, '', 'goals')
                    self.goals = goals
                elif t == ':metric':
                    self.metric = group.pop(0)
                else: self.parse_problem_extended(t, group)
        else:
            raise Exception('File ' + problem_filename + ' does not match problem pattern')

    def parse_problem_extended(self, t, group):
        print(str(t) + ' is not recognized in problem')

    #-----------------------------------------------
    # Split predicates
    #-----------------------------------------------

    def split_predicates(self, group, preds, name, part):
        if not type(group) is list:
            raise Exception('Error with ' + name + part)
        if group[0] == 'and':
            group.pop(0)
        else:
            group = [group]
        for predicate in group:
            if predicate[0] == 'not':
                if len(predicate) != 2:
                    raise Exception('Unexpected not in ' + name + part)
                preds.append(predicate)
            else:
                preds.append(predicate)

    #-----------------------------------------------
    # Groundify
    #-----------------------------------------------

    def groundify_vars(self, var_list, objects, types):
        if not var_list:
            yield []
            return
        type_map = []
        variables = []
        untyped_preds = []

        for v_name, predi in var_list.items():
            # print('\nP: ' + str(v_name) + ', ' + str(predi))
            ground_pred = [v_name]
            for var, type in predi.items():
                ground_pred.append(var)
                # print('var: ' + str(var) + '; type: ' + str(type))
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
            untyped_preds.append(ground_pred)
        # print(untyped_preds)
        for assignment in itertools.product(*type_map):
            # print(variables)
            # print(assignment)
            grounded_vars = self.replace(copy.deepcopy(untyped_preds), variables, assignment)
            # effects = self.replace(copy.deepcopy(self.effects), variables, assignment)
            yield grounded_vars
            # print(grounded_preconditions)

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

    # -----------------------------------------------
    # Instantiate Initial State
    # -----------------------------------------------



    def set_init_state(self):

        grounded_state_variables = {}
        for gps in self.groundify_vars(self.predicates, self.objects, self.types):
            for gpp in gps:
                grounded_state_variables[str(gpp)] = False
        for gfs in self.groundify_vars(self.functions, self.objects, self.types):
            for gf in gfs:
                grounded_state_variables[str(gf)] = 0.0

        self.init_state = State(state_vars=grounded_state_variables)
        self.init_state.instantiate(self.initialized_problem_state_variables)
        # return self.init_state

    # -----------------------------------------------
    # Ground all actions
    # -----------------------------------------------

    def set_grounded_actions(self):
        self.grounded_actions = []
        if constants.TEMPORAL_DOMAIN:
            self.grounded_actions.append(constants.TIME_PASSING_ACTION)

        for act in self.actions:
            for ga in act.groundify(self.objects, self.types):
                self.grounded_actions.append(ga)

    # -----------------------------------------------
    # Ground all events
    # -----------------------------------------------

    def set_grounded_events(self):
        self.grounded_events = []
        for eve in self.events:
            for ge in eve.groundify(self.objects, self.types):
                self.grounded_events.append(ge)

    # -----------------------------------------------
    # Ground all processes
    # -----------------------------------------------

    def set_grounded_processes(self):
        self.grounded_processes = []
        for pro in self.processes:
            for gp in pro.groundify(self.objects, self.types):
                self.grounded_processes.append(gp)

#-----------------------------------------------
# Main
#-----------------------------------------------
if __name__ == '__main__':
    import sys, pprint
    domain = sys.argv[1]
    problem = sys.argv[2]
    parser = PDDL_Parser(domain, problem)
    print('----------------------------')
    # pprint.pprint(parser.scan_tokens(domain))
    print('----------------------------')
    # pprint.pprint(parser.scan_tokens(problem))
    print('----------------------------')
    parser.parse_domain(domain)
    parser.parse_problem(problem)
    print('Domain name: ' + parser.domain_name)
    pprint.pprint(parser.predicates)
    pprint.pprint(parser.functions)
    for act in parser.actions:
        print(act)
    for eve in parser.events:
        print(eve)
    for pro in parser.processes:
        print(pro)
    print('----------------------------')
    print('Problem name: ' + parser.problem_name)
    print('Objects: ' + str(parser.objects))
    print('Types: ' + str(parser.types))
    print('Init State:')
    pprint.pprint(parser.initialized_problem_state_variables)
    print('Goals:')
    pprint.pprint(parser.goals)
    print('----------------------------')
