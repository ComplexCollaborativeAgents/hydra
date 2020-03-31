
import agent.consistency.model_repair as model_repair
import os.path as path
import settings
from agent.planning.pddlplus_parser import PddlProblemParser, PddlProblemExporter, PddlDomainParser, PddlDomainExporter
from agent.planning.model_manipulator import ManipulateInitNumericFluent
from agent.consistency.model_repair import *
import agent.planning.pddl_plus as pddl_plus

DATA_DIR = path.join(settings.ROOT_PATH, 'data')
GRAVITY = ["gravity"]

''' Helper function to get the gravity value from the initial state '''
def __get_gravity_value_in_init(pddl_problem):
    gravity_fluent = pddl_plus.get_numeric_fluent(pddl_problem.init, GRAVITY)
    return pddl_plus.get_numeric_fluent_value(gravity_fluent)


''' Test changing a single numeric fluent, and the ability of model repair to repair it'''
def test_single_numeric_repair():
    problem_file = path.join(DATA_DIR, "pddl_parser_test_problem.pddl")
    problem_parser = PddlProblemParser()
    pddl_problem = problem_parser.parse_pddl_problem(problem_file)
    assert pddl_problem is not None, "PDDL+ problem object not parsed" # Sanity check, parser works

    domain_file = path.join(DATA_DIR, "pddl_parser_test_domain.pddl")
    domain_parser = PddlDomainParser()
    pddl_domain = domain_parser.parse_pddl_domain(domain_file)
    assert pddl_domain is not None, "PDDL+ domain object not parsed" # Sanity check, parser works

    original_gravity_value = __get_gravity_value_in_init(pddl_problem)

    # Apply the change
    gravity_delta = 1
    manipulator = ManipulateInitNumericFluent(GRAVITY,3*gravity_delta)
    manipulator.apply_change(pddl_domain, pddl_problem)

    new_gravity_value = __get_gravity_value_in_init(pddl_problem)
    assert new_gravity_value, original_gravity_value+3*gravity_delta # Sanity check: manipulator working

    # Apply the model repair algorithm
    model_repair = SingleNumericFluentRepair(GRAVITY, gravity_delta)
    model_repair.repair(pddl_domain, pddl_problem)

    # Assert repair was able to restore the gravity to its correct value
    assert original_gravity_value == __get_gravity_value_in_init(pddl_problem)

