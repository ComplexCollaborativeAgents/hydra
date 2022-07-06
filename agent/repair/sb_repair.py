from agent.repair.focused_repair import *

logging.basicConfig(format='%(name)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("sb_repair")


class BirdLocationConsistencyEstimator(TimeIndependentConsistencyEstimator):
    """
    Checks consistency by considering the location of the birds 
    """

    def __init__(self, fluent_names=None, obs_prefix=100, discount_factor=0.9, consistency_threshold=20):
        if fluent_names is None:
            fluent_names = []
        super().__init__(fluent_names, obs_prefix, discount_factor, consistency_threshold)

    def consistency_from_trace(self, simulation_trace: list, state_seq: list, delta_t: float = DEFAULT_DELTA_T):
        """
        Estimate consistency by considering the location of the birds in the observed state seq
        """
        # Get birds
        birds = set()
        for [state, _, _] in simulation_trace:
            if len(birds) == 0:  # TODO: Discuss how to make this work. Currently a new bird suddenly appears
                birds = state.get_birds()
            else:
                break

        self.fluent_names = []
        for bird in birds:
            self.fluent_names.append(('x_bird', bird))
            self.fluent_names.append(('y_bird', bird))

        return TimeIndependentConsistencyEstimator.consistency_from_trace(self, simulation_trace, state_seq, delta_t)


class PigDeadConsistencyEstimator(ConsistencyEstimator):
    """
    Are the pigs we expected to be dead actually dead?
    """
    def __init__(self, fluent_names=None):
        if fluent_names is None:
            fluent_names = []
        super().__init__(fluent_names)

    def consistency_from_trace(self, simulation_trace: list, state_seq: list, delta_t: float = settings.SB_DELTA_T):
        """
        simulation_trace: list of (state, time, actions) lists in simulation (expected plan)
        state_seq: observations (what actually happened)
        delta_t: time step for simulation (useful, innit?)
        """
        live_pigs = 0
        last_state_in_obs = state_seq[-1]
        live_pigs_in_obs = last_state_in_obs.get_pigs()
        last_state_in_sim = simulation_trace[-1][0]
        pigs_in_sim = last_state_in_sim.get_pigs()
        for pig in pigs_in_sim:
            if last_state_in_sim.is_true(('pig_dead', pig)) and pig in live_pigs_in_obs:
                live_pigs += 50
        return live_pigs


class BlockNotDeadConsistencyEstimator(ConsistencyEstimator):
    """
    Checks consistency by considering which blocks are alive
    """
    MIN_BLOCK_NOT_DEAD = 50
    MAX_INCREMENT = 50

    def __init__(self, fluent_names=None):
        if fluent_names is None:
            fluent_names = []
        super().__init__(fluent_names)

    def consistency_from_trace(self, simulation_trace: list, state_seq: list, delta_t: float = settings.SB_DELTA_T):
        """
        Estimate consistency by considering the location of the birds in the observed state seq
        """
        # TODO: Next "if" statement should never be called but it is: check out why
        if len(state_seq) == 0:
            return 0
        last_state_in_obs = state_seq[-1]
        live_blocks_in_obs = last_state_in_obs.get_blocks()
        last_state_in_sim = simulation_trace[-1][0]
        blocks_in_sim = last_state_in_sim.get_blocks()
        # Make sure very block that is assumed to be dead in sim is indeed not alive in obs

        blocks_not_dead = 0.0
        for block in blocks_in_sim:
            life_fluent = ('block_life', block)
            life_value = last_state_in_sim[life_fluent]
            if life_value <= 0:
                if block in live_blocks_in_obs:
                    logger.debug("Block %s is alive but sim. thinks it is dead (life value=%.2f)" % (block, life_value))
                    blocks_not_dead = blocks_not_dead + 1

        # Rationale: having more blocks make the life of a block less predictable
        if blocks_not_dead > 0:
            return BlockNotDeadConsistencyEstimator.MIN_BLOCK_NOT_DEAD \
                   + blocks_not_dead / len(blocks_in_sim) * BlockNotDeadConsistencyEstimator.MAX_INCREMENT

        # TODO: Consider checking also the reverse condition
        return 0


class ExternalAgentLocationConsistencyEstimator(TimeIndependentConsistencyEstimator):
    """
    Checks consistency by considering the location of the birds
    """

    def __init__(self, fluent_names=None, obs_prefix=100, discount_factor=0.9, consistency_threshold=20):
        if fluent_names is None:
            fluent_names = []
        super().__init__(fluent_names, obs_prefix, discount_factor, consistency_threshold)

    ''' Estimate consistency by considering the location of the birds in the observed state seq '''

    def consistency_from_trace(self, simulation_trace: list, state_seq: list, delta_t: float = DEFAULT_DELTA_T):
        # Get birds
        agents = set()
        for [state, _, _] in simulation_trace:
            if len(agents) == 0:  # TODO: Discuss how to make this work. Currently a new bird suddenly appears
                agents = state.get_agents()
            else:
                break

        self.fluent_names = []
        for ag in agents:
            self.fluent_names.append(('x_agent', ag))
            self.fluent_names.append(('y_agent', ag))

        return TimeIndependentConsistencyEstimator.consistency_from_trace(self, simulation_trace, state_seq, delta_t)


class ScienceBirdsConsistencyEstimator(ConsistencyEstimator):
    """
    Checks consistency for SB
    """

    def __init__(self, fluent_names=None, use_simplified_problems=True, consistency_estimators=None):
        # ExternalAgentLocationConsistencyEstimator(), BirdLocationConsistencyEstimator()
        if fluent_names is None:
            fluent_names = []
        super().__init__(fluent_names)
        if consistency_estimators is None:
            consistency_estimators = [BirdLocationConsistencyEstimator(), BlockNotDeadConsistencyEstimator(),
                                      PigDeadConsistencyEstimator()]
        self.consistency_estimators = list()
        self.use_simplified_problems = use_simplified_problems
        self.consistency_estimators.extend(consistency_estimators)

    def traces_from_simulator(self, observation, meta_model: ScienceBirdsMetaModel,
                              simulator: PddlPlusSimulator = NyxPddlPlusSimulator(), delta_t: float = 0.025):
        """
        Computes the consistency of a given observation w.r.t the given meta model using the given simulator
        NOTICE: Using here the simplified problem due to SB domain's complexity
        """
        # TODO remove\rename\remodel this function appropriately.
        try:
            problem = meta_model.create_pddl_problem(observation.get_initial_state())
            if self.use_simplified_problems:
                problem = meta_model.create_simplified_problem(problem)
            domain = meta_model.create_pddl_domain(observation.get_initial_state())
            # domain = PddlPlusGrounder().ground_domain(domain, problem)  # Simulator accepts only grounded domains
            plan = observation.get_pddl_plan(meta_model)
            (_, _, expected_trace,) = simulator.simulate(plan, problem, domain, delta_t=delta_t)

            observed_seq = observation.get_pddl_states_in_trace(meta_model)
            consistency = self.consistency_from_trace(expected_trace, observed_seq, delta_t)
        except InconsistentPlanError as e:  # Sometimes the repair makes the executed plan be inconsistent, e.g., its preconditions are not satisfied
            consistency = ConsistencyEstimator.PLAN_FAILED_CONSISTENCY_VALUE
            logger.info(f'Could not compute consistency! {str(e)}')
        except KeyError as e:
            consistency = ConsistencyEstimator.PLAN_FAILED_CONSISTENCY_VALUE
            logger.info(f'Inconsistency calculator: No {str(e)} found, that is pretty inconsistent. ')
        except IndexError as e:
            consistency = 0
            logger.info('No observations to check, can not compute inconsistency. ')
        return consistency

    def consistency_from_trace(self, simulation_trace: list, state_seq: list, delta_t: float = DEFAULT_DELTA_T):
        max_inconsistency = 0
        for consistency_estimator in self.consistency_estimators:
            inconsistency = consistency_estimator.consistency_from_trace(simulation_trace, state_seq, delta_t)
            if inconsistency > max_inconsistency:
                max_inconsistency = inconsistency

        return max_inconsistency


class ScienceBirdsMetaModelRepair(GreedyBestFirstSearchMetaModelRepair):
    """ The meta model repair used for ScienceBirds. """
    # THIS CLASS ONLY USED IN TESTS
    def __init__(self, meta_model=ScienceBirdsMetaModel(),
                 consistency_threshold=settings.SB_CONSISTENCY_THRESHOLD,
                 time_limit=settings.SB_REPAIR_TIMEOUT,
                 max_iterations=settings.SB_REPAIR_MAX_ITERATIONS):
        constants_to_repair = meta_model.repairable_constants
        repair_deltas = meta_model.repair_deltas
        consistency_estimator = ScienceBirdsConsistencyEstimator()
        super().__init__(constants_to_repair, consistency_estimator, repair_deltas,
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
