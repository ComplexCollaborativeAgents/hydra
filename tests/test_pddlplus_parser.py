import os.path as path
import settings
import filecmp
from agent.planning.pddlplus_parser import PddlPlusDomain, PddlParser, PddlExporter, WorldChangeTypes

'''
Parses a PDDL+ domain file to objects and serializes it back to a PDDL+ file, then parse it again, serialize again, and check it is OK.
'''
def test_domain_parsing():
    pddl_file_name = "%s/sb_domain.pddl" % str(path.join(settings.ROOT_PATH, 'agent', 'planning', 'docker_scripts'))

    parser = PddlParser()
    pddl_domain = parser.parse_pddl_domain(pddl_file_name)
    assert pddl_domain is not None, "PDDL+ domain object fails"

    clone_file_name = "clone_sb_domain.pddl"

    exporter = PddlExporter()
    exporter.to_file(pddl_domain, clone_file_name)

    clone_pddl_domain = parser.parse_pddl_domain(clone_file_name)
    clone_file2_name = "clone_sb_domain2.pddl"
    exporter.to_file(clone_pddl_domain, clone_file2_name)

    assert (filecmp.cmp(clone_file_name, clone_file2_name, shallow=False))


'''
Parses a PDDL+ domain file to objects and serializes it back to a PDDL+ file, then parse it again, serialize again, and check it is OK.
'''
def test_domain_exporting():
    pddl_file_name = "pddl_parser_test_domain.pddl"

    parser = PddlParser()
    pddl_domain = parser.parse_pddl_domain(pddl_file_name)
    assert pddl_domain is not None, "PDDL+ domain object fails"

    clone_file_name = "pddl_parser_test_domain_clone.pddl"

    exporter = PddlExporter()
    exporter.to_file(pddl_domain, clone_file_name)

    clone_pddl_domain = parser.parse_pddl_domain(clone_file_name)
    clone_file2_name = "pddl_parser_test_domain_clone2.pddl"
    exporter.to_file(clone_pddl_domain, clone_file2_name)

    assert (filecmp.cmp(clone_file_name, clone_file2_name, shallow=False))

'''
Parses a PDDL+ domain file and checks the object 
'''
def test_domain_parsing():
    pddl_file_name = "pddl_parser_test_domain.pddl"

    parser = PddlParser()
    pddl_domain = parser.parse_pddl_domain(pddl_file_name)
    assert pddl_domain is not None, "PDDL+ domain object fails"

    assert pddl_domain.name == "angry_birds_scaled"
    assert len(pddl_domain.types) == 4
    assert len(pddl_domain.predicates) ==4
    assert len(pddl_domain.functions) == 22
    assert len(pddl_domain.processes) == 2
    assert len(pddl_domain.events) == 5
    assert len(pddl_domain.actions) == 1
