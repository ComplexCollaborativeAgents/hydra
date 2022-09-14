"""
Parses a PDDL+ file to objects and serializes it back to a PDDL+ file

Using code from https://norvig.com/lispy.html by Peter Norvig
"""
from enum import Enum
from agent.planning.pddl_plus import *


class PddlParserUtils:
    """
    A class with utility function to help parse PDDL files.
    """

    def tokenize(self, file_name: str) -> list:
        """ Converts the file in a list of tokens, considering space and newline as a delimiter,
                and considers each parenthesis as a token"""
        in_file = open(file_name, encoding='utf-8-sig')
        file_tokens = list()
        for line in in_file.readlines():
            if line.strip().startswith(";"):  # A comment line
                continue
            if ';' in line:
                line = line[:line.find(';')]  # strip away inline comment
            line_tokens = line.lower().replace("(", " ( ").replace(")", " ) ").split()
            for token in line_tokens:
                if len(token.strip()) == 0:
                    continue
                file_tokens.append(token)
        in_file.close()
        return file_tokens

    def read_from_tokens(self, tokens: list):
        """ A recursive function to create nodes in the parse tree"""
        if len(tokens) == 0:
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
            return token  # A basic atom in the syntax tree

    def parse_syntax_tree(self, file_name: str) -> list():
        """ Accepts a file written in LISP and outputs a syntax tree, in the form of a list of lists (recursively) """

        tokens = self.tokenize(file_name)
        return self.read_from_tokens(tokens)

    def write_tokens(self, tokens: list, out_file, prefix_str="", suffix_str="\n"):
        """ A recursive function to create nodes in the parse tree"""

        out_file.write("%s(" % prefix_str)
        first_token = True
        for token in tokens:
            if type(token) is list:
                if len(token) < 3:
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


class PddlDomainExporter:
    """
    Accepts a PddlPlusDomain object and outputs a PDDL+ file for the planner
    """

    def __init__(self):
        self.parse_utils = PddlParserUtils()

    def to_file(self, pddl_domain: PddlPlusDomain, output_file_name):
        """ Outputs this object to a PDDL file in a valid PDDL+ format, that can be run by UPMurphi """

        out_file = open(output_file_name, "w")
        out_file.write("(define(domain %s)\n" % pddl_domain.name)
        out_file.write(f"\t(:requirements {' '.join(pddl_domain.requirements)})\n")
        out_file.write("\t(:types %s)\n" % " ".join(pddl_domain.types))

        # Print predicates
        out_file.write("\t(:predicates\n")
        for predicate in pddl_domain.predicates:
            self.parse_utils.write_tokens(predicate, out_file, prefix_str="\t\t")
        out_file.write("\t)\n")

        # Print functions
        out_file.write("\t(:functions\n")
        for pddl_function in pddl_domain.functions:
            self.parse_utils.write_tokens(pddl_function, out_file, prefix_str="\t\t")
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

    def write_world_change(self, world_change: PddlPlusWorldChange, world_change_type, out_file):
        """ Write a process/event/action to the out file"""

        out_file.write("\t(:%s %s\n" % (world_change_type.name, world_change.name))
        if len(world_change.parameters) > 0:
            out_file.write("\t\t:parameters ")
            for parameter in world_change.parameters:
                self.parse_utils.write_tokens(parameter, out_file)

        if len(world_change.preconditions) > 0:
            out_file.write("\t\t:precondition (and \n")
            for precondition in world_change.preconditions:
                out_file.write("\t\t")
                self.parse_utils.write_tokens(precondition, out_file, prefix_str=" ")
            out_file.write("\t\t) \n")

        if len(world_change.effects) > 0:
            out_file.write("\t\t:effect (and \n")
            for effect in world_change.effects:
                out_file.write("\t\t")
                self.parse_utils.write_tokens(effect, out_file, prefix_str=" ")
            out_file.write("\t\t) \n")

        out_file.write("\t)  \n")


class PddlProblemExporter:
    """
    Accepts a PddlPlusProblem object and outputs a PDDL+ problem file for the planner
    """

    def to_file(self, pddl_problem: PddlPlusProblem, output_file_name):
        """ Outputs this object to a PDDL file in a valid PDDL+ format, that can be run by UPMurphi """

        parse_utils = PddlParserUtils()

        out_file = open(output_file_name, "w")
        out_file.write("(define(problem %s)\n" % pddl_problem.name)
        out_file.write("(:domain %s)\n" % pddl_problem.domain)

        # Print objects
        out_file.write("(:objects ")
        for object in pddl_problem.objects:
            out_file.write("%s - %s " % (object[0], object[1]))
        out_file.write(")\n")

        # Print init facts
        out_file.write("(:init ")
        for init_fact in pddl_problem.init:
            parse_utils.write_tokens(init_fact, out_file, prefix_str="\t", suffix_str="\n")
        out_file.write(")\n")

        out_file.write("(:goal (and ")
        for goal_fact in pddl_problem.goal:
            parse_utils.write_tokens(goal_fact, out_file, prefix_str=" ", suffix_str=" ")
        out_file.write("))\n")

        out_file.write("(:metric %s)\n" % pddl_problem.metric)

        out_file.write(")\n")
        out_file.close()


class PddlDomainParser():
    """
    Accepts a PDDL+ domain file and outputs a PddlPlusDomain object
    """

    def parse_types(self, element: list) -> list:
        """ Parses the types"""
        return element[1:]  # Ignore first element, which contains the :type string

    def parse_predicates(self, predicates_element: list) -> list:
        """ Parses the predicates"""
        return predicates_element[1:]

    def parse_functions(self, functions_element: list) -> list:
        """ Parses the functions"""
        return functions_element[1:]

    """ Parses the parameters of the process. The parameters start in index i.
    Returns the list of parameters and the index to the next element to parse in the process element"""

    def parse_world_change_parameters(self, i, world_change_element):
        i = i + 1  # To go after the :parameters string
        parameters = list()
        while i < len(world_change_element) and world_change_element[i][0].startswith(":") == False:
            parameters.append(world_change_element[i])
            i = i + 1
        return (i, parameters)

    def parse_world_change_preconditions(self, i, world_change_element):
        """ Parses the preconditions of the process. The preconditions start in index i.
            Returns the list of preconditions and the index to the next element to parse in the process element"""
        i = i + 1  # To go after the :parameters string
        preconditions_element = world_change_element[i]
        if preconditions_element[0] != "and":
            raise SyntaxError("Only supporting an (and) clause for preconditions")
        return (i + 1, preconditions_element[1:])

    def parse_world_change_effects(self, i, world_change_element):
        """ Parses the effects of the process. The effects start in index i.
                Returns the list of effects and the index to the next element to parse in the process element"""
        i = i + 1  # To go after the :parameters string
        effects_element = world_change_element[i]
        if effects_element[0] != "and":
            raise SyntaxError("Only supporting an (and) clause for effects")
        return (i + 1, effects_element[1:])

    def parse_world_change(self, world_change_element: list, world_change_type) -> list:
        """ Parses an element of type process, effect, or action"""
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

    def parse_pddl_domain(self, pddl_file_name: str) -> PddlPlusDomain:
        """
                Reads a PDDL+ file from the domain file and outputs a PDDL plus object
            """
        domain = PddlPlusDomain()
        parse_utils = PddlParserUtils()
        syntax_tree = parse_utils.parse_syntax_tree(pddl_file_name)

        assert (syntax_tree[0] == "define")  # Standard header of a PDDL domain file)

        for element in syntax_tree[1:]:
            if len(element) > 0:  # Element is a non-leaf
                if element[0] == "domain":
                    domain.name = element[1]
                elif element[0] == ":requirements":
                    domain.requirements = element[1:]
                elif element[0] == ":types":
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
                    domain.events.append(event)
                elif element[0] == ":action":
                    action = self.parse_world_change(element, WorldChangeTypes.action)
                    domain.actions.append(action)

        # print ("\n\nDOMAINNNNN: \n")
        # print (domain.name)
        # print (domain.types)
        # print (domain.predicates)
        # print (domain.functions)
        # for pro in domain.processes:
        #     pro.print_info()
        # print ("")
        # for ev in domain.events:
        #     ev.print_info()
        # print ("")
        # for ac in domain.actions:
        #     ac.print_info()

        return domain


class PddlProblemParser():
    """
Accepts a PDDL+ problem file and outputs a PddlPlusProblem object
"""

    def parse_objects(self, element: list) -> list:
        """ Parses the objects. Objects are in the format object_name - object_type"""
        i = 1
        objects = list()
        while i + 2 < len(element):
            objects.append((element[i], element[i + 2]))
            assert element[i + 1].strip() == "-"
            i = i + 3
        return objects

    def parse_init(self, element: list) -> list:
        """ Parses the initial state list of facts"""
        return element[1:]

    def parse_goal(self, element):
        """ Parses the goal condition """
        # Asserting the current focus is on conjunctive goals, i.e., the goal is an AND over a set of facts
        assert len(element) == 2
        assert element[1][0] == "and"
        return element[1][1:]

    def parse_metric(self, element: list) -> list:
        """ Parses the metric function"""
        # Asserting a single metric
        assert len(element) == 3

        return "%s(%s)" % (element[1], element[2][0])  # Metric is f(x), which parsed to two tokens: f and x

    def parse_pddl_problem(self, pddl_file_name: str) -> PddlPlusDomain:
        """
                Reads a PDDL+ file from the problem file and outputs a PDDL plus proble object
            """
        problem = PddlPlusProblem()
        parse_utils = PddlParserUtils()
        syntax_tree = parse_utils.parse_syntax_tree(pddl_file_name)

        assert (syntax_tree[0] == "define")  # Standard header of a PDDL domain file)

        for element in syntax_tree[1:]:
            if len(element) > 0:  # Element is a non-leaf
                if element[0] == "problem":
                    problem.name = element[1]
                elif element[0] == ":domain":
                    problem.domain = element[1]
                elif element[0] == ":objects":
                    problem.objects.extend(self.parse_objects(element))
                elif element[0] == ":init":
                    problem.init = self.parse_init(element)
                elif element[0] == ":goal":
                    problem.goal = self.parse_goal(element)
                elif element[0] == ":metric":
                    problem.metric = self.parse_metric(element)
        return problem
