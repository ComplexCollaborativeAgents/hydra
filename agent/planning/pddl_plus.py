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
    def __str__(self):
        return str(self.name)


class PddlPlusProblem():
    def __init__(self):
        self.name = None
        self.domain = None
        self.objects = list()
        self.init = list()
        self.goal = list()
        self.metric = None

    def __eq__(self, other):
        if isinstance(other, PddlPlusProblem) == False:
            return False
        if self.name!=other.name:
            return False
        if self.domain!=other.domain:
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
        if self.metric!=other.metric:
            return False
        return True


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

    def get_action(self,action_name):
        for action in self.actions:
            if action.name==action_name:
                return action
        return None

'''
A class representing a PDDL+ state. Contains only the fluents and their values. 
'''
class PddlPlusState():
    ''' Creates a PDDL+ state object initialized by a list of fluent given in the PddlPlusProblem format of lists'''
    def __init__(self, fluent_list: list = None):
        self.numeric_fluents = dict()
        self.boolean_fluents = set()
        if fluent_list is not None:
            self.load_from_fluent_list(fluent_list)

    def __getitem__(self, fluent_name):
        if fluent_name in self.numeric_fluents:
            return self.get_value(fluent_name)
        elif fluent_name in self.boolean_fluents:
            return self.is_true(fluent_name)
        else:
            return False # For the case of Boolean fluents, this makes sense



    ''' Loads the PddlPlusState object with a list of tuples as imported from PDDL file. 
        A tuple representing a numeric variable is of the form (=, object_list, value). 
        A tuple representing a boolean variable is of the form (object_list) or (not object_list)'''
    def load_from_fluent_list(self, fluent_list:list):
        for fluent in fluent_list:
            if fluent[0]=="=": # This is a numeric fluent
                fluent_name = tuple(fluent[1])
                self.numeric_fluents[fluent_name]=float(fluent[2]) # Wrapping in tuple to be hashable, converting to float (not string)
            else: # This is a boolean fluent
                if fluent[0]!="not":
                    fluent_name = tuple(fluent)
                    self.boolean_fluents.add(fluent_name) # Wrapping in tuple to be hashable

    ''' Returns this state as a list of fluents, compatible with PddlProblem'''
    def save_as_fluent_list(self):
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

    ''' Returns the set of bird objects alive in this state. 
     Bird is identified by the x_bird fluent. Returns a set of bird names. '''
    def get_birds(self):
        birds = set()
        for fluent_name in self.numeric_fluents:
            # We expect every bird has an x coordinate in a fluent of the form (x_bird, birdname)
            if len(fluent_name)==2 and fluent_name[0]=="x_bird":
                birds.add(fluent_name[1])
        return birds

    ''' Returns the active bird'''
    def get_active_bird(self):
        active_bird_id = int(self['active_bird'])
        return self.get_bird(active_bird_id)

    ''' Get the bird with the given bird id'''
    def get_bird(self, bird_id: int):
        for bird in self.get_birds(): # TODO: Can change this to be more efficient
            if bird_id == int(self[("bird_id", bird)]):
                return bird
        raise ValueError("Bird %d not found in state" % bird_id)

    ''' Returns the set of bird objects alive in this state. 
     Bird is identified by the x_bird fluent. Returns a set of bird names. '''
    def get_pigs(self):
        pigs = set()
        for fluent_name in self.numeric_fluents:
            # We expect every bird has an x coordinate in a fluent of the form (x_bird, birdname)
            if len(fluent_name)==2 and fluent_name[0]=="x_pig":
                pigs.add(fluent_name[1])
        return pigs

    # Deep compare
    def __eq__(self, other):
        if isinstance(other, PddlPlusState)==False:
            return False
        if self.numeric_fluents!=other.numeric_fluents:
            return False
        if self.boolean_fluents!=other.boolean_fluents:
            return False
        return True

    # String representation
    def __str__(self):
        string_buffer = ""
        for fluent_name in self.numeric_fluents:
            string_buffer = "%s %s=%s\n" %  (string_buffer, str(fluent_name), self.numeric_fluents[fluent_name])
        for fluent_name in self.boolean_fluents:
            string_buffer = "%s %s\n" % (string_buffer, str(fluent_name))
        return string_buffer

    ''' Export as a string in PDDL (lisp) format '''
    def to_pddl(self):
        string_buffer = ""
        for fluent_name in self.numeric_fluents:
            string_buffer = "%s (=%s %s)\n" %  (string_buffer, str(fluent_name), self.numeric_fluents[fluent_name])
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

        grounded_domain.events.extend(self.__ground_world_change_lists(domain.events, problem))
        grounded_domain.processes.extend(self.__ground_world_change_lists(domain.processes, problem))
        grounded_domain.actions.extend(self.__ground_world_change_lists(domain.actions, problem))

        return grounded_domain

    ''' Ground a list of world_change objects to the given problem '''
    def __ground_world_change_lists(self, world_change_list, problem):
        grounded_world_change_list = list()
        for world_change in world_change_list:
            assert len(world_change.parameters)==1
            world_change_parameters = world_change.parameters[0]
            process_parameters = self.__get_typed_parameter_list(world_change_parameters)
            all_bindings = self.__get_possible_bindings(process_parameters, problem)
            for binding in all_bindings:
                grounded_world_change_list.append(self.ground_world_change(world_change, binding))
        return grounded_world_change_list

    ''' Extracts from a raw list of the form [?x - typex y? - type?] a list of the form [(?x typex)(?y typey)] 
    TODO: Think about where this really should go '''
    def __get_typed_parameter_list(self, element: list):
        i = 0
        typed_parameters = list()
        while i < len(element):
            assert element[i].startswith("?")  # A parameter
            assert element[i + 1] == "-"
            typed_parameters.append((element[i], element[i + 2]))  # lifted object name and type
            i = i + 3
        return typed_parameters

    ''' Extract the list of typed parameters of the given predict'''
    def __get_predicate_parameters(self, predicate):
        parameter_list = predicate[1:]
        return self.__get_typed_parameter_list(parameter_list)

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
    def __init__(self, actions: list = list()):
        for action in actions:
            if isinstance(action, TimedAction)==False:
                raise ValueError("Action %s is not a TimedAction or a [action,time] pair" % action) # This check should probably be removed at some stage
            self.append(action)

    ''' Adds a list of [[action_name, action_time]...]. Converts actions to appropriate WorldChange objects '''
    def add_raw_actions(self, raw_timed_action_list, grounded_domain: PddlPlusDomain):
        for raw_timed_action in raw_timed_action_list:
            action_name  = raw_timed_action[0]
            action = grounded_domain.get_action(action_name)
            self.append(TimedAction(action, float(raw_timed_action[1])))

''' Check if a given string is a float. TODO: Replace this with a more elegant python way of doing this.'''
def is_float( text :str ):
    try:
        float(text)
        return True
    except:
        return False

