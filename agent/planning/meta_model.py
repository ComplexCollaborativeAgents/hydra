
from agent.planning.pddlplus_parser import *

class MetaModel():
    ''' A meta-model used to generated PDDL+ domains and problems from observations '''

    def __init__(self,
                 docker_path: str,
                 domain_file_name: str,
                 delta_t,
                 metric: str,
                 repairable_constants :list,
                 repair_deltas: list = None,
                 constant_numeric_fluents:dict = {},
                 constant_boolean_fluents:dict = {}):

        self.docker_path = docker_path
        self.domain_file_name = domain_file_name
        self.delta_t = delta_t
        self.hyper_parameters = dict() # These are parameters that do not appear in the PDDL files

        self.constant_numeric_fluents = dict(constant_numeric_fluents)
        self.constant_boolean_fluents = dict(constant_boolean_fluents)

        self.repairable_constants = list(repairable_constants)
        if repair_deltas is None:
            self.repair_deltas = [1.0] * len(self.repairable_constants)
        else:
            self.repair_deltas = repair_deltas

        self.metric = metric

    def create_pddl_problem(self, observed_state) -> PddlPlusProblem:
        ''' Transtlate the observed state to a PDDL+ probelm for this domain in which the observed state is the initial state '''
        raise NotImplementedError()

    def create_pddl_state(self, observed_state) -> PddlPlusState:
        ''' Transtlate the observed state to a PDDL+ state in this domain '''
        raise NotImplementedError()

    def create_pddl_domain(self, observed_state) -> PddlPlusDomain:
        ''' Create a PDDL+ domain for the given observed state '''
        domain_file = "{}/{}".format(str(self.docker_path), self.domain_file_name)
        domain_parser = PddlDomainParser()
        return domain_parser.parse_pddl_domain(domain_file)