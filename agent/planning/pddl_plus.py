'''
Utility file for handling PDDL+ domains and problems.

Using code from https://norvig.com/lispy.html by Peter Norvig
'''
from enum import Enum



'''
Return a numeric fluent from the list of fluents
'''
def get_numeric_fluent(fluent_list, fluent_name):
    for fluent in fluent_list:
        if fluent[0]=="=": # Note: I intentionally not added an AND with the next if, to not fall for cases where len(fluent)==1
            if fluents_names_equals(fluent[1], fluent_name):
                return fluent
    raise ValueError("Fluent %s not found in list" % fluent_name)

'''
Return the value of a given numeric fluent name
'''
def get_numeric_fluent_value(fluent):
    return fluent[-1]


'''
A fluent is defined by a identifier and a set of objects. 
Two fluent names are equal if these are equal.  
'''
def fluents_names_equals(fluent_name1, fluent_name2):
    if len(fluent_name1)!=len(fluent_name2):
        return False
    for i in range(len(fluent_name1)):
        if fluent_name1[i]!=fluent_name2[i]:
            return False
    return True

'''
Returns true if fluent is a numeric value
'''
def is_numeric(fluent):
    if is_numeric(fluent[-1]):
        return True
    else:
        return False



class WorldChangeTypes(Enum):
    process = 1
    event = 2
    action = 3

''' A class that represents an event, process, or action'''
class PddlPlusWorldChange():
    def __init__(self, type : WorldChangeTypes):
        self.name = None
        self.type = type
        self.parameters = list()
        self.preconditions = list()
        self.effects = list()


class PddlPlusProblem():
    def __init__(self):
        self.name = None
        self.domain = None
        self.objects = list()
        self.init = list()
        self.goal = list()
        self.metric = None


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

'''
A class representing a PDDL+ state. Contains only the fluents and their values. 
'''
class PddlPlusState():
    ''' Creates a PDDL+ state object initialized by a list of fluent given in the PddlPlusProblem format of lists'''
    def __init__(self, fluent_list: list = None):
        self.numeric_fluents = dict()
        self.boolean_fluents = set()
        if fluent_list is not None:
            self.load(fluent_list)

    ''' Loads the PddlPlusState object with a list of tuples as imported from PDDL file. 
        A tuple representing a numeric variable is of the form (=, object_list, value). 
        A tuple representing a boolean variable is of the form (object_list) or (not object_list)'''
    def load(self, fluent_list:list):
        for fluent in fluent_list:
            if fluent[0]=="=": # This is a numeric fluent
                fluent_name = tuple(fluent[1])
                self.numeric_fluents[fluent_name]=fluent[2] # Wrapping in tuple to be hashable
            else: # This is a boolean fluent
                if fluent[0]!="not":
                    fluent_name = tuple(fluent)
                    self.boolean_fluents.add(fluent_name) # Wrapping in tuple to be hashable

    def is_true(self, boolean_fluent_name):
        if boolean_fluent_name in self.boolean_fluents:
            return True
        else:
            return False

    def get_value(self, numeric_fluent_name):
        return self.numeric_fluents[numeric_fluent_name]

    # Deep compare
    def __eq__(self, other):
        if isinstance(other, PddlPlusState)==False:
            return False
        if self.numeric_fluents!=other.numeric_fluents:
            return False
        if self.boolean_fluents!=other.boolean_fluents:
            return False
        return True

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




''' Class responsible for all groundings'''
class PddlPlusGrounder():
    ''' Recursively ground the given element with the given binding '''

    def ground_element(self, element, binding):
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
        new_name = world_change.name
        for parameter in world_change.parameters:
            assert parameter[0] in binding  # Asserts all the parameters are bound
            new_name = "%s %s" % (new_name, binding[parameter[0]])
            # TODO: Chech that binding respects types.
        grounded_world_change.name = new_name

        for precondition in world_change.preconditions:
            grounded_world_change.preconditions.append(self.ground_element(precondition, binding))
        for effect in world_change.effects:
            grounded_world_change.effects.append(self.ground_element(effect, binding))

        return grounded_world_change

    ''' Created a grounded version of this domain '''
    def ground_domain(self, domain: PddlPlusDomain, problem : PddlPlusProblem):

        grounded_domain = PddlPlusDomain()
        grounded_domain.name = domain.name
        grounded_domain.types = domain.types # TODO: Probably unnecessary

        for predicate in domain.predicates:
            predicate_parameters = self.__get_predicate_parameters(predicate)
            if len(predicate_parameters)==0:
                grounded_domain.predicates.append(predicate)
            else:
                all_bindings = self.__get_possible_bindings(predicate_parameters, problem)
                for binding in all_bindings:
                    grounded_domain.predicates.append(self.ground_element(predicate, binding))

        for process in domain.processes:
            all_bindings = self.__get_possible_bindings(process.parameters, problem)
            for binding in all_bindings:
                grounded_domain.processes.append(self.ground_world_change(process,binding))

        for event in domain.events:
            all_bindings = self.__get_possible_bindings(event.parameters, problem)
            for binding in all_bindings:
                grounded_domain.events.append(self.ground_world_change(event,binding))

        for action in domain.actions:
            all_bindings = self.__get_possible_bindings(action.parameters, problem)
            for binding in all_bindings:
                grounded_domain.actions.append(self.ground_world_change(action,binding))

        return grounded_domain

    ''' Extract the list of typed parameters of the given predict'''
    def __get_predicate_parameters(self, predicate):
        i = 0
        typed_parameters=list()
        while i<len(predicate):
            if predicate[i].startswith("?"): # A parameter
                assert predicate[i+1]=="-"
                typed_parameters.append((predicate[i], predicate[i+2])) # lifted object name and type
                i = i+3
            else:
                i = i+1
        return typed_parameters


    ''' Enumerate all possible bindings of the given parameters to the given problem '''
    def __get_possible_bindings(self, parameters, problem : PddlPlusProblem):
        all_bindings = list()
        self.__recursive_get_possible_bindings(parameters, problem, dict(), all_bindings)
        return all_bindings

    ''' Recursive method to find all bindings '''
    def __recursive_get_possible_bindings(self, parameters, problem, binding, bindings):
        for object in problem.objects:
            if self.__can_bind(parameters[0], object):
                assert parameters[0][0].startswith("?")

                binding[parameters[0][0]]=object[0]
                if len(parameters) == 1:
                    bindings.append(binding.copy())
                else:
                    self.__recursive_get_possible_bindings(parameters[1:], problem, binding, bindings)

    ''' Checks if one can bound the given parameter to the given object '''
    def __can_bind(self, parameter, object):
        return object[-1]==parameter[-1]

''' An action with a time stamp saying when it should start'''
class TimedAction():
    def __init__(self, action: PddlPlusWorldChange, start_at : float):
        self.action = action
        self.start_at = start_at

''' Just a list of timed actions '''
class PddlPlusPlan(list):
    def __init__(self, actions: list):
        for action in actions:
            if isinstance(action, TimedAction)==False:
                raise ValueError("Action %s is not a TimedAction" % action) # This check should probably be removed at some stage
            self.append(action)

