'''
Parses a PDDL+ file to objects and serializes it back to a PDDL+ file

Using code from https://norvig.com/lispy.html by Peter Norvig
'''
from enum import Enum

class PddlPlusDomain():
    def __init__(self):
        self.name = None
        self.types = list()
        self.predicates = list()
        self.functions = list()
        self.constants = list()
        self.processes = list()
        self.actions = list()
        self.events = list()


class WorldChangeTypes(Enum):
    process = 1
    event = 2
    action = 3

''' Processes, events, and actions are all things that change the state of the world'''
class PddlPlusWorldChange():
    def __init__(self, type : WorldChangeTypes):
        self.name = None
        self.type = type
        self.parameters = list()
        self.preconditions = list()
        self.effects = list()


'''
Accepts a PddlPlusDomain object and outputs a PDDL+ file for the planner
'''
class PddlExporter():
    ''' Outputs this object to a PDDL file in a valid PDDL+ format, that can be run by UPMurphi '''
    def to_file(self, pddl_domain:PddlPlusDomain, output_file_name):

        out_file = open(output_file_name, "w")
        out_file.write("(define(domain %s)\n" % pddl_domain.name)
        out_file.write(
            "\t(:requirements :typing :durative-actions :duration-inequalities :fluents :time :negative-preconditions :timed-initial-literals)\n")
        out_file.write("\t(:types %s)\n" % " ".join(pddl_domain.types))

        # Print predicates
        out_file.write("\t(:predicates\n")
        for predicate in pddl_domain.predicates:
            self.write_tokens(predicate, out_file, prefix_str = "\t\t")
        out_file.write("\t)\n")

        # Print functions
        out_file.write("\t(:functions\n")
        for pddl_function in pddl_domain.functions:
            self.write_tokens(pddl_function, out_file, prefix_str = "\t\t")
        out_file.write("\t)\n")

        # Print processes
        for process in pddl_domain.processes:
            self.write_world_change(process, WorldChangeTypes.process, out_file)

        # Print events
        for event in pddl_domain.events:
            self.write_world_change(event, WorldChangeTypes.event, out_file)

        # Print actions
        for action in pddl_domain.actions:
            self.write_world_change(action, WorldChangeTypes.action, out_file)

        out_file.write(")\n")
        out_file.close()

    ''' Write a process/event/action to the out file'''
    def write_world_change(self, world_change:PddlPlusWorldChange, world_change_type, out_file):
        out_file.write("\t(:%s %s\n" % (world_change_type.name, world_change.name))
        if len(world_change.parameters)>0:
            out_file.write("\t\t:parameters ")
            for parameter in world_change.parameters:
                self.write_tokens(parameter, out_file)

        if len(world_change.preconditions)>0:
            out_file.write("\t\t:precondition (and \n")
            for precondition in world_change.preconditions:
                out_file.write("\t\t")
                self.write_tokens(precondition, out_file, prefix_str = " ")
            out_file.write("\t\t) \n")

        if len(world_change.effects)>0:
            out_file.write("\t\t:effect (and \n")
            for effect in world_change.effects:
                out_file.write("\t\t")
                self.write_tokens(effect, out_file, prefix_str = " ")
            out_file.write("\t\t) \n")

        out_file.write("\t)  \n")


    ''' A recursive function to create nodes in the parse tree'''
    def write_tokens(self, tokens: list, out_file, prefix_str = "", suffix_str="\n"):
        out_file.write("%s(" % prefix_str)
        first_token = True
        for token in tokens:
            if type(token) is list:
                if len(token)<3:
                    self.write_tokens(token, out_file, prefix_str=" ", suffix_str=" ")
                else:
                    self.write_tokens(token, out_file, prefix_str)
            else:
                if first_token == False:
                    out_file.write(" %s" % token)
                else:
                    out_file.write("%s" % token)
            first_token = False

        out_file.write(")%s" % suffix_str)


'''
Accepts a PDDL+ file and outputs a PddlPlusDomain object
'''
class PddlParser():

    ''' Converts the file in a list of tokens, considering space and newline as a delimiter,
        and considers each parenthesis as a token'''
    def tokenize(self, file_name:str) -> list:
        in_file = open(file_name, encoding='utf-8-sig')
        file_tokens = list()
        for line in in_file.readlines():
            if line.strip().startswith(";"): # A comment line
                continue
            line_tokens = line.replace("(", " ( ").replace(")"," ) ").split()
            for token in line_tokens:
                if len(token.strip())==0:
                    continue
                file_tokens.append(token)
        in_file.close()
        return file_tokens

    ''' A recursive function to create nodes in the parse tree'''
    def read_from_tokens(self, tokens: list):
        if len(tokens)==0:
            raise SyntaxError("Unexpected EOF")
        token = tokens.pop(0)

        if token == '(':
            node = []
            while tokens[0] != ')':
                node.append(self.read_from_tokens(tokens))

            tokens.pop(0)  # pop off ')'
            return node
        elif token == ')':
            raise SyntaxError('unexpected )')
        else:
            return token # A basic atom in the syntax tree

    ''' Accepts a file written in LISP and outputs a syntax tree, in the form of a list of lists (recursively) '''
    def parse_syntax_tree(self, file_name: str) -> list():
        tokens = self.tokenize(file_name)
        return self.read_from_tokens(tokens)


    ''' Parses the types'''
    def parse_types(self, element: list) -> list:
        return element[1:] # Ignore first element, which contains the :type string

    ''' Parses the predicates'''
    def parse_predicates(self, predicates_element: list) -> list:
        return predicates_element[1:]


    ''' Parses the functions'''
    def parse_functions(self, predicates_element: list) -> list:
        return predicates_element[1:]

    ''' Parses the parameters of the process. The parameters start in index i. 
    Returns the list of parameters and the index to the next element to parse in the process element'''
    def parse_world_change_parameters(self, i, process_element):
        i = i+1 # To go after the :parameters string
        parameters = list()
        while i < len(process_element) and process_element[i][0].startswith(":")==False:
            parameters.append(process_element[i])
            i=i+1
        return (i, parameters)

    ''' Parses the preconditions of the process. The preconditions start in index i. 
        Returns the list of preconditions and the index to the next element to parse in the process element'''
    def parse_world_change_preconditions(self, i, process_element):
        i = i + 1  # To go after the :parameters string
        preconditions_element = process_element[i]
        if preconditions_element[0]!="and":
            raise SyntaxError("Only supporting an (and) clause for preconditions")
        return (i+1, preconditions_element[1:])

    ''' Parses the effects of the process. The effects start in index i. 
        Returns the list of effects and the index to the next element to parse in the process element'''
    def parse_world_change_effects(self, i, process_element):
        i = i + 1  # To go after the :parameters string
        effects_element = process_element[i]
        if effects_element[0] != "and":
            raise SyntaxError("Only supporting an (and) clause for effects")
        return (i + 1, effects_element[1:])

    ''' Parses an element of type process, effect, or action'''
    def parse_world_change(self, world_change_element: list, world_change_type) -> list:
        world_change = PddlPlusWorldChange(world_change_type)

        world_change.name = world_change_element[1]  # element[0] is the :process string
        i = 2
        while i < len(world_change_element):
            if world_change_element[i] == ":parameters":
                (i, parameters) = self.parse_world_change_parameters(i, world_change_element)
                world_change.parameters = parameters
            if world_change_element[i] == ":precondition":
                (i, preconditions) = self.parse_world_change_preconditions(i, world_change_element)
                world_change.preconditions = preconditions
            if world_change_element[i] == ":effect":
                (i, effects) = self.parse_world_change_effects(i, world_change_element)
                world_change.effects = effects
        return world_change

    '''
        Reads a PDDL+ file from the domain file and outputs a PDDL plus object
    '''
    def parse_pddl_domain(self, pddl_file_name: str) -> PddlPlusDomain:
        domain = PddlPlusDomain()
        syntax_tree = self.parse_syntax_tree(pddl_file_name)

        assert(syntax_tree[0]=="define") # Standard header of a PDDL domain file)

        for element in syntax_tree[1:]:
            if len(element)>0: # Element is a non-leaf
                if element[0] == "domain":
                    domain.name = element[1]
                elif element[0]==":types":
                    domain.types = self.parse_types(element)
                elif element[0] == ":predicates":
                    domain.predicates = self.parse_predicates(element)
                elif element[0] == ":functions":
                    domain.functions = self.parse_functions(element)
                elif element[0] == ":process":
                    process = self.parse_world_change(element, WorldChangeTypes.process)
                    domain.processes.append(process)
                elif element[0] == ":event":
                    event = self.parse_world_change(element, WorldChangeTypes.event)
                    domain.events.append(process)
                elif element[0] == ":action":
                    action = self.parse_world_change(element, WorldChangeTypes.action)
                    domain.actions.append(action)
        return domain