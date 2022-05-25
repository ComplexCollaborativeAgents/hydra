#!/usr/bin/env python
# Four spaces as indentation [no tabs]

import itertools
import re

from agent.planning.nyx.compiler import JIT
from agent.planning.nyx.compiler.preconditions_tree import PreconditionsTree
from agent.planning.nyx.syntax.action import Action
from agent.planning.nyx.syntax.event import Event
from agent.planning.nyx.syntax.process import Process
from agent.planning.nyx.syntax.state import State
import agent.planning.nyx.syntax.constants as constants


class PDDLDomain:

    def __init__(self, **kwargs):
        self.name = kwargs.get('name', None)
        self.requirements = kwargs.get('requirements', None)
        self.types = kwargs.get('types', dict())
        self.predicates = kwargs.get('predicates', dict())
        self.functions = kwargs.get('functions', dict())
        self.constants = kwargs.get('constants', dict())
        self.processes = kwargs.get('processes', list())
        self.actions = kwargs.get('actions', list())
        self.events = kwargs.get('events', list())


class PDDLProblem:

    def __init__(self, **kwargs):
        self.name = kwargs.get('name', None)
        self.init = kwargs.get('init', None)
        self.objects = kwargs.get('objects', dict())
        self.goals = kwargs.get('goals', list())
        self.metric = kwargs.get('metric', None)


class GroundedPDDLInstance:

    def __init__(self, domain: PDDLDomain, problem: PDDLProblem):
        self.domain = domain
        self.problem = problem

        self.enact_requirements()
        self._objects = None
        self._initialize_state()
        self._goals_code, self.goals = JIT.compile_expression(self.problem.goals, name='goals')
        if self.problem.metric != ['total-time'] and self.problem.metric != ['total-actions'] and self.problem.metric is not None:
            self._metric_code, self.metric = JIT.compile_expression([self.problem.metric], name='metric')
        self.processes = self._groundify_happenings(self.domain.processes)
        self.events = self._groundify_happenings(self.domain.events)
        self.actions = self._groundify_happenings(self.domain.actions)
        if constants.TEMPORAL_DOMAIN:
            self.actions.add_happening(constants.TIME_PASSING_ACTION)

        # with open('grounded_domain.txt', 'w') as gd_file:
        #     gd_file.write(str(self))

    def enact_requirements(self):
        for requirement in self.domain.requirements:
            if requirement == ':time':
                constants.TEMPORAL_DOMAIN = True

    @property
    def objects(self) -> dict:
        if self._objects is None:
            self._objects = dict()
            for obj_source in [self.domain.constants, self.problem.objects]:
                for type_name, obj_list in obj_source.items():
                    if type_name not in self._objects:
                        self._objects[type_name] = []
                    self._objects[type_name].extend((obj for obj in obj_list if obj not in self._objects[type_name]))
        return self._objects

    @staticmethod
    def _groundify(variables: dict, objects: dict):
        grounded_vars = []
        for var_name, type_pairs in variables.items():
            grounded_type_instances = [objects[type_name] for _, type_name in type_pairs.items()] # if objects.get(type_name)]
            # Only ground objects of a type if there are any objects of that type.
            for almost_grounded in itertools.product(*grounded_type_instances):
                grounded_vars.append([var_name] + list(almost_grounded))
        return grounded_vars

    def _initialize_state(self):
        state_variables = {}
        for grounded_predicate in self._groundify(self.domain.predicates, self.objects):
            state_variables[str(grounded_predicate)] = False
        for grounded_function in self._groundify(self.domain.functions, self.objects):
            state_variables[str(grounded_function)] = 0.0
        self.init_state = State(state_vars=state_variables)
        self.init_state.instantiate(self.problem.init)

    def _groundify_happenings(self, happenings) -> PreconditionsTree:
        preconditions_tree = PreconditionsTree()
        for happening in happenings:
            for grounded_happening in happening.groundify(self.objects, self.domain.types):
                preconditions_tree.add_happening(grounded_happening)
        return preconditions_tree

    def __str__(self):
        res = 'Domain: ' + self.domain.name + '\n'
        res += 'Problem' + self.problem.name + '\n'
        res += 'Initial state: ' + str(self.init_state) + '\n'
        res += 'actions: \n'
        for action in self.actions:
            res += action.name + '\npreconditions: ' + str(action.preconditions) + '\neffects: ' + str(action.effects) + '\n'
        res += 'events: \n'
        for event in self.events:
            res += event.name + '\npreconditions: ' + str(event.preconditions) + '\neffects: ' + str(event.effects) + '\n'
        res += 'proccesses: \n'
        for process in self.processes:
            res += process.name + '\npreconditions: ' + str(process.preconditions) + '\neffects: ' + str(process.effects) + '\n'
        return res


class PDDL_Parser:

    SUPPORTED_REQUIREMENTS = [':strips', ':adl', ':negative-preconditions', ':typing', ':time', ':fluents', ':continuous-effects', ':disjunctive-preconditions', ':semantic-attachment']

    def __init__(self, domain_file, problem_file):
        self.domain = PDDLDomain()
        self.problem = PDDLProblem()
        self.parse_domain(domain_file)
        self.parse_problem(problem_file)
        self.grounded_instance = GroundedPDDLInstance(self.domain, self.problem)

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
            self.domain.name = 'unknown'
            while tokens:
                group = tokens.pop(0)
                t = group.pop(0)
                if t == 'domain':
                    self.domain.name = group[0]
                elif t == ':requirements':
                    for req in group:
                        if req == ':time':
                            constants.TEMPORAL_DOMAIN = True
                        if req == ':semantic-attachment':
                            # (NOT AVAILABLE YET ON MASTER BRANCH, hidden feature for now)
                            constants.SEMANTIC_ATTACHMENT = True
                            # raise Exception('Requirement ' + req + ' not officially supported yet!')
                        if not req in self.SUPPORTED_REQUIREMENTS:
                            raise Exception('Requirement ' + req + ' not supported')
                    self.domain.requirements = group
                elif t == ':constants':
                    self.parse_constants(group, t)
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
    # Parse constants
    #-----------------------------------------------

    def parse_constants(self, group, name):
        self.parse_hierarchy(group, self.domain.constants, name, False)

    #-----------------------------------------------
    # Parse objects
    #-----------------------------------------------

    def parse_objects(self, group, name):
        self.parse_hierarchy(group, self.problem.objects, name, False)

    # -----------------------------------------------
    # Parse types
    # -----------------------------------------------

    def parse_types(self, group):
        self.parse_hierarchy(group, self.domain.types, 'types', True)

    #-----------------------------------------------
    # Parse predicates
    #-----------------------------------------------

    def parse_predicates(self, group):
        for pred in group:
            predicate_name = pred.pop(0)
            if predicate_name in self.domain.predicates:
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
            self.domain.predicates[predicate_name] = arguments

    # -----------------------------------------------
    # Parse functions
    # -----------------------------------------------

    def parse_functions(self, group):
        for func in group:
            function_name = func.pop(0)
            if function_name in self.domain.functions:
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
            self.domain.functions[function_name] = arguments

    #-----------------------------------------------
    # Parse action
    #-----------------------------------------------

    def parse_action(self, group):
        name = group.pop(0)
        if not type(name) is str:
            raise Exception('Action without name definition')
        for act in self.domain.actions:
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
        self.domain.actions.append(Action(name, parameters, preconditions, effects))

    def parse_action_extended(self, t, group):
        print(str(t) + ' is not recognized in action')

    # -----------------------------------------------
    # Parse event
    # -----------------------------------------------

    def parse_event(self, group):
        name = group.pop(0)
        if not type(name) is str:
            raise Exception('Event without name definition')
        for eve in self.domain.events:
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
        self.domain.events.append(Event(name, parameters, preconditions, effects, extensions))

    def parse_event_extended(self, t, group):
        print(str(t) + ' is not recognized in event')

    # -----------------------------------------------
    # Parse process
    # -----------------------------------------------

    def parse_process(self, group):
        name = group.pop(0)
        if not type(name) is str:
            raise Exception('Process without name definition')
        for pro in self.domain.processes:
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
        self.domain.processes.append(Process(name, parameters, preconditions, effects, extensions))

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
            self.problem.name = 'unknown'
            while tokens:
                group = tokens.pop(0)
                t = group.pop(0)
                if t == 'problem':
                    self.problem.name = group[0]
                elif t == ':domain':
                    if self.domain.name != group[0]:
                        raise Exception('Different domain specified in problem file')
                elif t == ':requirements':
                    pass # Ignore requirements in problem, parse them in the domain
                elif t == ':objects':
                    self.parse_objects(group, t)
                elif t == ':init':
                    # print('\nREAD INIT STATE\n')
                    # pprint.pprint(group)
                    self.problem.init = group
                elif t == ':goal':
                    goals = []
                    self.split_predicates(group[0], goals, '', 'goals')
                    self.problem.goals = goals
                elif t == ':metric':
                    self.problem.metric = group.pop(0)
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
# Main
#-----------------------------------------------
if __name__ == '__main__':
    import sys, pprint
    domain = sys.argv[1]
    problem = sys.argv[2]
    parser = PDDL_Parser(domain, problem)
    print('----------------------------')
    print('Domain name: ' + parser.domain.name)
    pprint.pprint(parser.domain.predicates)
    pprint.pprint(parser.domain.functions)
    for act in parser.domain.actions:
        print(act)
    for eve in parser.domain.events:
        print(eve)
    for pro in parser.domain.processes:
        print(pro)
    print('----------------------------')
    print('Problem name: ' + parser.problem.name)
    print('Objects: ' + str(parser.grounded_instance.objects))
    print('Types: ' + str(parser.domain.types))
    print('Init State:')
    pprint.pprint(parser.problem.init)
    print('Goals:')
    pprint.pprint(parser.problem.goals)
    print('----------------------------')
