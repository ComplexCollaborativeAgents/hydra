from agent.consistency.pddl_plus_simulator import *

''' Helper function: simulate the given plan, on the given problem and domain.  '''
def simulate_plan_trace(plan: PddlPlusPlan, problem:PddlPlusProblem, domain: PddlPlusDomain, delta_t:float = 0.05):
    simulator = PddlPlusSimulator()
    grounded_domain = PddlPlusGrounder().ground_domain(domain, problem)  # Simulator accepts only grounded domains
    (_, _, trace) =  simulator.simulate(plan, problem, grounded_domain, delta_t)
    return trace

''' Helper function: returns a PDDL+ problem and domain objects'''
def load_problem_and_domain(problem_file_name :str, domain_file_name: str):
    parser = PddlDomainParser()
    pddl_domain = parser.parse_pddl_domain(domain_file_name)
    assert pddl_domain is not None, "PDDL+ domain object not parsed"

    parser = PddlProblemParser()
    pddl_problem = parser.parse_pddl_problem(problem_file_name)
    assert pddl_problem is not None, "PDDL+ problem object not parsed"

    return (pddl_problem, pddl_domain)

''' Setup the logger for the given module name'''
def create_logger(logger_name: str):
    logging.basicConfig(format='%(name)s - %(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    return logger