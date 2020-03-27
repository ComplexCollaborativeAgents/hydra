import os.path as path
import settings
import filecmp
from agent.planning.pddlplus_parser import PddlPlusDomain, PddlDomainParser, PddlDomainExporter, WorldChangeTypes, PddlProblemParser, PddlPlusProblem, PddlProblemExporter


DATA_DIR = path.join(settings.ROOT_PATH, 'data')

'''
Parses a PDDL+ domain file to objects and serializes it back to a PDDL+ file, then parse it again, serialize again, and check it is OK.
'''
def test_domain_parsing():
    pddl_file_name = path.join(DATA_DIR, "pddl_parser_test_domain.pddl")

    parser = PddlDomainParser()
    pddl_domain = parser.parse_pddl_domain(pddl_file_name)
    assert pddl_domain is not None, "PDDL+ domain object fails"

    clone_file_name = "clone_sb_domain.pddl"

    exporter = PddlDomainExporter()
    exporter.to_file(pddl_domain, clone_file_name)

    clone_pddl_domain = parser.parse_pddl_domain(clone_file_name)
    clone_file2_name = "clone_sb_domain2.pddl"
    exporter.to_file(clone_pddl_domain, clone_file2_name)

    assert (filecmp.cmp(clone_file_name, clone_file2_name, shallow=False))


'''
Parses a PDDL+ domain file to objects and serializes it back to a PDDL+ file, then parse it again, serialize again, and check it is OK.
'''
def test_domain_exporting():
    pddl_file_name = path.join(DATA_DIR, "pddl_parser_test_domain.pddl")

    parser = PddlDomainParser()
    pddl_domain = parser.parse_pddl_domain(pddl_file_name)
    assert pddl_domain is not None, "PDDL+ domain object fails"

    clone_file_name = path.join(DATA_DIR, "pddl_parser_test_domain_clone.pddl")

    exporter = PddlDomainExporter()
    exporter.to_file(pddl_domain, clone_file_name)

    clone_pddl_domain = parser.parse_pddl_domain(clone_file_name)
    clone_file2_name = path.join(DATA_DIR, "pddl_parser_test_domain_clone2.pddl")
    exporter.to_file(clone_pddl_domain, clone_file2_name)

    assert (filecmp.cmp(clone_file_name, clone_file2_name, shallow=False))

'''
Parses a PDDL+ domain file and checks the object 
'''
def test_domain_parsing():
    pddl_file_name = path.join(DATA_DIR, "pddl_parser_test_domain.pddl")

    parser = PddlDomainParser()
    pddl_domain = parser.parse_pddl_domain(pddl_file_name)
    assert pddl_domain is not None, "PDDL+ domain object empty"

    assert pddl_domain.name == "angry_birds_scaled"
    assert len(pddl_domain.types) == 4
    assert len(pddl_domain.predicates) ==4
    assert len(pddl_domain.functions) == 22
    assert len(pddl_domain.processes) == 2
    assert len(pddl_domain.events) == 5
    assert len(pddl_domain.actions) == 1


'''
Parses a PDDL+ domain file and checks the object 
'''
def test_problem_parsing():
    pddl_file_name = path.join(DATA_DIR, "pddl_parser_test_problem.pddl")

    parser = PddlProblemParser()
    pddl_problem = parser.parse_pddl_problem(pddl_file_name)
    assert pddl_problem is not None, "PDDL+ domain object fails"

    assert pddl_problem.name == "angry_birds_prob"
    assert pddl_problem.domain == "angry_birds_scaled"
    assert len(pddl_problem.objects) == 7
    assert len(pddl_problem.init) ==26
    assert len(pddl_problem.goal) == 1
    assert pddl_problem.metric == "minimize(total-time)"

''' 
Parse a PDDL+ problem file, export it, then parse it again, export again, and compare 
'''
def test_problem_exporting():
    pddl_file_name = path.join(DATA_DIR, "pddl_parser_test_problem.pddl")

    parser = PddlProblemParser()
    pddl_problem = parser.parse_pddl_problem(pddl_file_name)
    assert pddl_problem is not None, "PDDL+ problem empty"

    clone_file_name = path.join(DATA_DIR, "pddl_parser_test_problem_clone.pddl")

    exporter = PddlProblemExporter()
    exporter.to_file(pddl_problem, clone_file_name)

    clone_pddl_problem = parser.parse_pddl_problem(clone_file_name)
    clone_file2_name = path.join(DATA_DIR, "pddl_parser_test_problem_clone2.pddl")
    exporter.to_file(clone_pddl_problem, clone_file2_name)

    assert (filecmp.cmp(clone_file_name, clone_file2_name, shallow=False))


