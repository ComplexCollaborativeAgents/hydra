from os import path
import settings
from agent.planning.pddlplus_parser import PddlProblemParser, PddlProblemExporter
from agent.planning.model_manipulator import ManipulateInitNumericFluent
import agent. planning.pddl_plus as pddl_plus

DATA_DIR = path.join(settings.ROOT_PATH, 'data')


def get_numeric_fluent(fluent_list, fluent_name):
    """
    Return a numeric fluent from the list of fluents
    """
    for fluent in fluent_list:
        if fluent[0] == "=":
            # Note: I intentionally not added an AND with the next if, to not fall for cases where len(fluent)==1
            # Yoni: But Python has lazy evaluation, Roni. What are you doing?? Also, this code seems wrong.
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

'''
    Load a Pddl domain and problem files. 
    Manipulate one of the problem's single-valued functions
    See if it works
'''
def test_problem_manipulation():
    pddl_file_name = path.join(DATA_DIR, "pddl_parser_test_problem.pddl")

    parser = PddlProblemParser()
    pddl_problem = parser.parse_pddl_problem(pddl_file_name)
    assert pddl_problem is not None, "PDDL+ problem object not parsed"

    # Manipulate gravity
    GRAVITY = ["gravity"]
    gravity_delta = 1
    gravity_fluent = get_numeric_fluent(pddl_problem.init, GRAVITY)
    original_gravity_value = float(get_numeric_fluent_value(gravity_fluent))

    manipulator = ManipulateInitNumericFluent(GRAVITY, gravity_delta)
    manipulator.apply_change(None, pddl_problem)
    gravity_fluent = get_numeric_fluent(pddl_problem.init, GRAVITY)
    modified_gravity_value = float(get_numeric_fluent_value(gravity_fluent))

    assert float(modified_gravity_value)==float(original_gravity_value)+float(gravity_delta)

    # Export to file
    exporter = PddlProblemExporter()
    clone_file_name = "clone_sb_domain.pddl"
    exporter.to_file(pddl_problem, clone_file_name)

    # Import clone and original, compare gravities
    original_problem = parser.parse_pddl_problem(pddl_file_name)
    original_gravity_fluent = get_numeric_fluent(original_problem.init, GRAVITY)
    modified_problem = parser.parse_pddl_problem(clone_file_name)
    modified_gravity_fluent = get_numeric_fluent(modified_problem.init, GRAVITY)
    assert float(get_numeric_fluent_value(modified_gravity_fluent)) \
           == float(get_numeric_fluent_value(original_gravity_fluent))+gravity_delta

