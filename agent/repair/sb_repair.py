import settings
from agent.repair.focused_repair import *

logging.basicConfig(format='%(name)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("sb_repair")


class PigDeadConsistencyEstimator(AspectConsistency):
    """
    Are the pigs we expected to be dead actually dead?
    """

    def __init__(self, fluent_names=None):
        if fluent_names is None:
            fluent_names = []
        super().__init__(fluent_names, fluent_template='pig')

    def consistency_from_trace(self, simulation_trace: list, state_seq: list, delta_t: float = settings.SB_DELTA_T):
        """
        simulation_trace: list of (state, time, actions) lists in simulation (expected plan)
        state_seq: observations (what actually happened)
        delta_t: time step for simulation (useful, innit?)
        """
        pigs_not_dead = self._objects_in_last_frame(simulation_trace, state_seq, in_sim=True, in_obs=False)
        return pigs_not_dead * 50


class BlockNotDeadConsistencyEstimator(AspectConsistency):
    """
    Checks consistency by considering which blocks are alive
    """
    MIN_BLOCK_NOT_DEAD = 50
    MAX_INCREMENT = 50

    def __init__(self, fluent_names=None):
        if fluent_names is None:
            fluent_names = []
        super().__init__(fluent_names, fluent_template='block')

    def consistency_from_trace(self, simulation_trace: list, state_seq: list, delta_t: float = settings.SB_DELTA_T):
        """
        Estimate consistency by considering which blocks are or aren't destroyed.
        """
        last_state_in_sim = simulation_trace[-1][0]
        blocks_in_sim = last_state_in_sim.get_blocks()
        blocks_not_dead = self._objects_in_last_frame(simulation_trace, state_seq, in_sim=False, in_obs=True)
        return BlockNotDeadConsistencyEstimator.MIN_BLOCK_NOT_DEAD \
               + blocks_not_dead / len(blocks_in_sim) * BlockNotDeadConsistencyEstimator.MAX_INCREMENT


class ExternalAgentLocationConsistencyEstimator(AspectConsistency):
    """
    Checks consistency by considering the location of the birds
    """

    def __init__(self):
        super().__init__([], 'agent')

    def consistency_from_trace(self, simulation_trace: list, state_seq: list, delta_t: float = DEFAULT_DELTA_T):
        """ Estimate consistency by considering the location of the external agents in the observed state seq """
        return self._trajectory_compare(simulation_trace, state_seq, delta_t)


class BirdLocationConsistencyEstimator(AspectConsistency):
    """
    Checks consistency by considering the location of the birds
    """

    def __init__(self):
        super().__init__([], 'bird')

    def consistency_from_trace(self, simulation_trace: list, state_seq: list, delta_t: float = DEFAULT_DELTA_T):
        """
        Estimate consistency by considering the location of the birds in the observed state seq
        """
        return self._trajectory_compare(simulation_trace, state_seq, delta_t)


class ScienceBirdsConsistencyEstimator(DomainConsistency):
    """
    Checks consistency for SB
    """

    def __init__(self, use_simplified_problems=True):
        # ExternalAgentLocationConsistencyEstimator(), BirdLocationConsistencyEstimator()
        super().__init__([BirdLocationConsistencyEstimator(), BlockNotDeadConsistencyEstimator(),
                          PigDeadConsistencyEstimator()])
        self.use_simplified_problems = use_simplified_problems

    def get_traces_from_simulator(self, observation, meta_model, simulator: PddlPlusSimulator, delta_t):
        problem = meta_model.create_pddl_problem(observation.get_initial_state())
        if self.use_simplified_problems:
            problem = meta_model.create_simplified_problem(problem)
        domain = meta_model.create_pddl_domain(observation.get_initial_state())
        # domain = PddlPlusGrounder().ground_domain(domain, problem)  # Simulator accepts only grounded domains
        plan = observation.get_pddl_plan(meta_model)
        (_, _, expected_trace,) = simulator.simulate(plan, problem, domain, delta_t=delta_t)
        observed_seq = observation.get_pddl_states_in_trace(meta_model)
        return expected_trace, observed_seq

    def consistency_from_simulator(self, observation, meta_model: ScienceBirdsMetaModel,
                                   simulator: PddlPlusSimulator = NyxPddlPlusSimulator(),
                                   delta_t: float = settings.SB_DELTA_T):
        """
        Computes the consistency of a given observation w.r.t the given meta model using the given simulator
        NOTICE: Using here the simplified problem due to SB domain's complexity
        """
        # TODO remove\rename\remodel this function appropriately.
        try:
            expected_trace, observed_seq = self.get_traces_from_simulator(observation, meta_model, simulator, delta_t)
            consistency = self.consistency_from_trace(expected_trace, observed_seq, delta_t)
        except InconsistentPlanError as e:
            # Sometimes the repair makes the executed plan be inconsistent, e.g., its preconditions are not satisfied
            consistency = AspectConsistency.PLAN_FAILED_CONSISTENCY_VALUE
            logger.info(f'Could not compute consistency! {str(e)}')
        except KeyError as e:
            consistency = AspectConsistency.PLAN_FAILED_CONSISTENCY_VALUE
            logger.info(f'Inconsistency calculator: No {str(e)} found, that is pretty inconsistent. ')
        except IndexError as e:
            consistency = 0
            logger.info('No observations to check, can not compute inconsistency. ')
        return consistency


class ScienceBirdsMetaModelRepair(GreedyBestFirstSearchContantFluentMetaModelRepair):
    """ The meta model repair used for ScienceBirds. """

    # THIS CLASS ONLY USED IN TESTS
    def __init__(self, meta_model=ScienceBirdsMetaModel(),
                 consistency_threshold=settings.SB_CONSISTENCY_THRESHOLD,
                 time_limit=settings.SB_REPAIR_TIMEOUT,
                 max_iterations=settings.SB_REPAIR_MAX_ITERATIONS):
        constants_to_repair = meta_model.repairable_constants
        repair_deltas = meta_model.repair_deltas
        consistency_estimator = ScienceBirdsConsistencyEstimator()
        super().__init__(meta_model, consistency_estimator,constants_to_repair,  repair_deltas,
                         consistency_threshold=consistency_threshold,
                         max_iterations=max_iterations,
                         time_limit=time_limit)

# class ScienceBirdsFocusedMetaModelRepair(FocusedMetaModelRepair):
#     THIS CLASS NOT USED ANYWHERE
#     def __init__(self, meta_model=ScienceBirdsMetaModel(),
#                  consistency_threshold=settings.SB_CONSISTENCY_THRESHOLD,
#                  time_limit=settings.SB_REPAIR_TIMEOUT,
#                  max_iterations=settings.SB_REPAIR_MAX_ITERATIONS):
#         constants_to_repair = meta_model.repairable_constants
#         repair_deltas = meta_model.repair_deltas
#         consistency_estimator = ScienceBirdsConsistencyEstimator()
#         super().__init__(constants_to_repair, consistency_estimator, repair_deltas,
#                          consistency_threshold=consistency_threshold,
#                          max_iterations=max_iterations,
#                          time_limit=time_limit)
