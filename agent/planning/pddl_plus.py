"""
Utility file for handling PDDL+ domains and problems.

Using code from https://norvig.com/lispy.html by Peter Norvig
"""
from enum import Enum
from collections import defaultdict


def get_numeric_fluent(fluent_list, fluent_name):
    """
    Return a numeric fluent from the list of fluents
    """
    for fluent in fluent_list:
        if fluent[0] == "=":
            # Note: I intentionally not added an AND with the next if, to not fall for cases where len(fluent)==1
            # Yoni: But Pythong has lazy evaluation, Roni. What are you doing?? Also, this code seems wrong.
            if fluents_names_equals(fluent[1], fluent_name):
                return fluent
    raise ValueError("Fluent %s not found in list" % fluent_name)


def get_numeric_fluent_value(fluent):
    """
    Return the value of a given numeric fluent name
    """
    return fluent[-1]


def fluents_names_equals(fluent_name1, fluent_name2):
    """
    A fluent is defined by a identifier and a set of objects.
    Two fluent names are equal if these are equal.
    """
    if len(fluent_name1) != len(fluent_name2):
        return False
    for i in range(len(fluent_name1)):
        if fluent_name1[i] != fluent_name2[i]:
            return False
    return True


class WorldChangeTypes(Enum):
    process = 1
    event = 2
    action = 3


class PddlPlusWorldChange:
    """ A class that represents an event, process, or action"""

    def __init__(self, type: WorldChangeTypes):
        self.name = None
        self.type = type
        self.parameters = list()
        self.preconditions = list()
        self.effects = list()

    def print_info(self):
        print("\n\nNAME:", self.name)
        print("TYPE:", self.type)
        print("\tPARAMS: ", end=" ")
        for par in self.parameters:
            print("(", par, ")", end=" ")
        print("\n\tPRECOND: ", end=" ")
        for prec in self.preconditions:
            print("(", prec, ")", end=" ")
        print("\n\tEFFECTS: ", end=" ")
        for eff in self.effects:
            print("(", eff, ")", end=" ")

    def __str__(self):
        return str(self.name)

    __repr__ = __str__


class PddlPlusProblem:
    def __init__(self):
        self.name = None
        self.domain = None
        self.objects = list()
        self.init = list()  # TODO make this a PDDLState instead of a list
        self.goal = list()
        self.metric = None

    def __eq__(self, other):
        if not isinstance(other, PddlPlusProblem):
            return False
        if self.name != other.name:
            return False
        if self.domain != other.domain:
            return False
        for object in self.objects:
            if object not in other.objects:
                return False
        for object in other.objects:
            if object not in self.objects:
                return False
        for init_item in self.init:
            if init_item not in other.init:
                return False
        for init_item in other.init:
            if init_item not in self.init:
                return False
        for goal_item in self.goal:
            if goal_item not in other.goal:
                return False
        for goal_item in other.goal:
            if goal_item not in self.goal:
                return False
        if self.metric != other.metric:
            return False
        return True

    def get_init_state(self):
        """ Get the initial state in PDDL+ format"""
        return PddlPlusState(self.init)


class PddlPlusDomain:
    def __init__(self):
        self.name = None
        self.requirements = list()
        self.types = list()
        self.predicates = list()
        self.functions = list()
        self.constants = list()
        self.processes = list()
        self.actions = list()
        self.events = list()

    def get_action(self, action_name):
        for action in self.actions:
            # print ("COMPARE: ", action.name, " vs ", action_name)
            if action.name.lower() == action_name.lower():
                return action
        print("\nNO MATCHING ACTIONS IN LIST:", self.actions)
        return None


class PddlPlusState:
    """
    A class representing a PDDL+ state. Contains only the fluents and their values.
    """

    def __init__(self, fluent_list: list = None):
        """ Creates a PDDL+ state object initialized by a list of fluent given in the PddlPlusProblem format of lists"""
        self.numeric_fluents = defaultdict(
            lambda: 0.0)  # Default value of non-existing fluents is zero in current planner.
        self.boolean_fluents = set()
        if fluent_list is not None:
            self.load_from_fluent_list(fluent_list)

    def __getitem__(self, fluent_name):
        if fluent_name in self.numeric_fluents:
            return self.get_value(fluent_name)
        elif fluent_name in self.boolean_fluents:
            return self.is_true(fluent_name)
        else:
            return False  # For the case of Boolean fluents, this makes sense

    def __contains__(self, fluent_name):
        if fluent_name in self.numeric_fluents:
            return True
        if fluent_name in self.boolean_fluents:
            return True
        return False

    def load_from_fluent_list(self, fluent_list: list):
        """ Loads the PddlPlusState object with a list of tuples as imported from PDDL file.
        A tuple representing a numeric variable is of the form (=, object_list, value).
        A tuple representing a boolean variable is of the form (object_list) or (not object_list)"""
        for fluent in fluent_list:
            if fluent[0] == "=":  # This is a numeric fluent
                fluent_name = tuple(fluent[1])
                self.numeric_fluents[fluent_name] = float(fluent[2])
                # Wrapping in tuple to be hashable, converting to float (not string)
            else:  # This is a boolean fluent
                if fluent[0] != "not":
                    fluent_name = tuple(fluent)
                    self.boolean_fluents.add(fluent_name)  # Wrapping in tuple to be hashable

    def save_as_fluent_list(self):
        """ Returns this state as a list of fluents, compatible with PddlProblem"""
        fluent_list = []
        for fluent_name in self.numeric_fluents:
            fluent_value = self.get_value(fluent_name)
            fluent_name_as_str = "(%s)" % (",".join(fluent_name))
            fluent_list.append(["=", fluent_name_as_str, fluent_value])
        for fluent_name in self.boolean_fluents:
            fluent_name_as_str = "(%s)" % (",".join(fluent_name))
            fluent_list.append(fluent_name_as_str)
        return fluent_list

    def is_true(self, boolean_fluent_name):
        if boolean_fluent_name in self.boolean_fluents:
            return True
        else:
            return False

    def get_value(self, numeric_fluent_name):
        if numeric_fluent_name not in self.numeric_fluents:
            assert False
        return self.numeric_fluents[numeric_fluent_name]

    def get_objects(self, name):
        """
        Gets all objects of a given Science bird type, e.g. bird, agent, platform or pig.
        """
        # TODO this blongs is a SB specific sub-class, not here
        objects = set()
        for fluent_name in self.numeric_fluents:
            # We expect every bird has an x coordinate in a fluent of the form (x_bird, birdname)
            if len(fluent_name) == 2 and fluent_name[0] == "x_" + name:
                objects.add(fluent_name[1])
        return objects

    def get_birds(self):
        # TODO this blongs is a SB specific sub-class, not here
        """ Returns the set of bird objects alive in this state.
             Bird is identified by the x_bird fluent. Returns a set of bird names. """
        return self.get_objects('bird')

    def get_agents(self):
        # TODO this blongs is a SB specific sub-class, not here
        return self.get_objects('agent')


    def get_active_bird(self):
        """ Returns the active bird"""
        # TODO this blongs is a SB specific sub-class, not here
        active_bird_id = int(self['active_bird'])
        return self.get_bird(active_bird_id)


    def get_bird(self, bird_id: int):
        """ Get the bird with the given bird id"""
        # TODO this blongs is a SB specific sub-class, not here
        for bird in self.get_birds():  # TODO: Can change this to be more efficient
            if bird_id == int(self[("bird_id", bird)]):
                return bird
        raise ValueError("Bird %d not found in state" % bird_id)


    def get_pigs(self):
        """ Returns the set of bird objects alive in this state.
             Bird is identified by the x_bird fluent. Returns a set of bird names. """
        # TODO this blongs is a SB specific sub-class, not here
        return self.get_objects('pig')


    def get_platforms(self):
        """ Returns the set of bird objects alive in this state.
             Bird is identified by the x_bird fluent. Returns a set of bird names. """
        # TODO this blongs is a SB specific sub-class, not here
        return self.get_objects('platform')


    def get_blocks(self):
        """ Returns the set of block objects present in this state.
             Block is identified by the x_block fluent. Returns a set of block names. """
        # TODO this blongs is a SB specific sub-class, not here
        return self.get_objects('block')

    # Deep compare
    def __eq__(self, other):
        if isinstance(other, PddlPlusState) == False:
            return False
        if self.numeric_fluents != other.numeric_fluents:
            return False
        if self.boolean_fluents != other.boolean_fluents:
            return False
        return True

    # String representation
    def __str__(self):
        string_buffer = ""
        for fluent_name in self.numeric_fluents:
            string_buffer = "%s %s=%s\n" % (string_buffer, str(fluent_name), self.numeric_fluents[fluent_name])
        for fluent_name in self.boolean_fluents:
            string_buffer = "%s %s\n" % (string_buffer, str(fluent_name))
        return string_buffer

    def diff(self, other):
        result = {}
        for fluent, value in self.numeric_fluents.items():
            if other.numeric_fluents.get(fluent) != value:
                result[fluent] = f'self: {value}, other: {other.numeric_fluents.get(fluent)}'
        for fluent in self.boolean_fluents:
            if fluent not in other.boolean_fluents:
                result[fluent] = f'Boolean fluent not in other'
        return result



    def to_pddl(self):
        """ Export as a string in PDDL (lisp) format """
        string_buffer = ""
        for fluent_name in self.numeric_fluents:
            string_buffer = "%s (=%s %s)\n" % (string_buffer, str(fluent_name), self.numeric_fluents[fluent_name])
        for fluent_name in self.boolean_fluents:
            string_buffer = "%s %s\n" % (string_buffer, str(fluent_name))
        return string_buffer

    # Printing capabilities for debug purposes
    def to_print(self):
        for fluent_name in self.numeric_fluents:
            print("%s=%s" % (str(fluent_name), self.numeric_fluents[fluent_name]))
        for fluent_name in self.boolean_fluents:
            print("%s" % str(fluent_name))

    # Deep clone
    def clone(self):
        new_state = PddlPlusState()
        for numeric_fluent_name in self.numeric_fluents:
            new_state.numeric_fluents[numeric_fluent_name] = self.numeric_fluents[numeric_fluent_name]
        for boolean_fluent in self.boolean_fluents:
            new_state.boolean_fluents.add(boolean_fluent)
        return new_state




class PddlPlusGrounder:
    # TODO is this still in use?
    """ Class responsible for all groundings"""

    def __init__(self, no_dummy_objects=False):
        self.no_dummy_objects = no_dummy_objects


    def ground_element(self, element, binding):
        """ Recursively ground the given element with the given binding """
        if isinstance(element, list):
            grounded_element = list()
            for sub_element in element:
                grounded_element.append(self.ground_element(sub_element, binding))
        else:
            assert isinstance(element, str)
            if element in binding:
                grounded_element = binding[element]
            else:
                grounded_element = element
        return grounded_element

    def ground_world_change(self, world_change: PddlPlusWorldChange, binding: dict):
        grounded_world_change = PddlPlusWorldChange(world_change.type)

        new_name = "%s %s" % (world_change.name, " ".join([value for value in binding.values()]))

        grounded_world_change.name = new_name

        for precondition in world_change.preconditions:
            grounded_world_change.preconditions.append(self.ground_element(precondition, binding))
        for effect in world_change.effects:
            grounded_world_change.effects.append(self.ground_element(effect, binding))

        return grounded_world_change


    def ground_domain(self, domain: PddlPlusDomain, problem: PddlPlusProblem):
        """ Created a grounded version of this domain """

        grounded_domain = PddlPlusDomain()
        grounded_domain.name = domain.name
        grounded_domain.types = domain.types  # TODO: Probably unnecessary

        for predicate in domain.predicates:
            predicate_parameters = self.__get_predicate_parameters(predicate)
            if len(predicate_parameters) == 0:
                grounded_domain.predicates.append(predicate)
            else:
                all_bindings = self.__get_possible_bindings(predicate_parameters, problem)
                for binding in all_bindings:
                    grounded_domain.predicates.append(self.ground_element(predicate, binding))

        grounded_domain.events.extend(self.__ground_world_change_lists(domain.events, problem))
        grounded_domain.processes.extend(self.__ground_world_change_lists(domain.processes, problem))
        grounded_domain.actions.extend(self.__ground_world_change_lists(domain.actions, problem))

        return grounded_domain

    def __ground_world_change_lists(self, world_change_list, problem):
        """ Ground a list of world_change objects to the given problem """
        grounded_world_change_list = list()
        for world_change in world_change_list:
            assert len(world_change.parameters) == 1
            world_change_parameters = world_change.parameters[0]
            process_parameters = self.__get_typed_parameter_list(world_change_parameters)
            all_bindings = self.__get_possible_bindings(process_parameters, problem)
            for binding in all_bindings:
                grounded_world_change_list.append(self.ground_world_change(world_change, binding))
        return grounded_world_change_list


    def __get_typed_parameter_list(self, element: list):
        """
        Extracts from a raw list of the form [?x - typex y? - type?] a list of the form [(?x typex)(?y typey)]
        TODO: Think about where this really should go
        """
        i = 0
        typed_parameters = list()
        while i < len(element):
            assert element[i].startswith("?")  # A parameter
            assert element[i + 1] == "-"
            typed_parameters.append((element[i], element[i + 2]))  # lifted object name and type
            i = i + 3
        return typed_parameters

    def __get_predicate_parameters(self, predicate):
        """ Extract the list of typed parameters of the given predict"""
        parameter_list = predicate[1:]
        return self.__get_typed_parameter_list(parameter_list)


    def __get_possible_bindings(self, parameters, problem: PddlPlusProblem):
        """ Enumerate all possible bindings of the given parameters to the given problem """
        all_bindings = list()
        self.__recursive_get_possible_bindings(parameters, problem, dict(), all_bindings)
        return all_bindings


    def __recursive_get_possible_bindings(self, parameters, problem, binding, bindings):
        """ Recursive method to find all bindings """
        for object in problem.objects:
            if len(object) > 1 and self.no_dummy_objects and "dummy" in object[0]:
                continue
            if self.__can_bind(parameters[0], object):
                assert parameters[0][0].startswith("?")

                binding[parameters[0][0]] = object[0]
                if len(parameters) == 1:
                    bindings.append(binding.copy())
                else:
                    self.__recursive_get_possible_bindings(parameters[1:], problem, binding, bindings)


    def __can_bind(self, parameter, object):
        """ Checks if one can bound the given parameter to the given object """
        return object[-1] == parameter[-1]



class TimedAction:
    """ An action with a time stamp saying when it should start"""
    def __init__(self, action_name: str, start_at: float):
        self.action_name = action_name
        self.start_at = round(start_at, 8)

    def __str__(self):
        return "t=%s, %s" % (self.start_at, self.action_name)



class PddlPlusPlan(list):
    """ Just a list of timed actions """
    def __init__(self, actions=None):
        super().__init__()
        if actions is None:
            actions = list()
        for action in actions:
            if isinstance(action, TimedAction) == False:
                raise ValueError(
                    "Action %s is not a TimedAction or a [action,time] pair" % action)  # This check should probably be removed at some stage
            self.append(action)



def is_float(text: str):
    """ Check if a given string is a float. TODO: Replace this with a more elegant python way of doing this."""
    try:
        float(text)
        return True
    except:
        return False


def is_op(op_name: str):
    """ Check if the given string is one of the supported mathematical operations """

    if op_name in ("+-/*=><"):
        return True
    elif op_name == "<=" or op_name == ">=":
        return True
    else:
        return False
