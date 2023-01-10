import logging

import settings
from agent.consistency.consistency_estimator import DomainConsistency, AspectConsistency
from agent.consistency.nyx_pddl_simulator import NyxPddlPlusSimulator
from agent.consistency.pddl_plus_simulator import PddlPlusSimulator, InconsistentPlanError
from agent.planning.sb_meta_model import ScienceBirdsMetaModel
from agent.planning.pddl_plus import PddlPlusGrounder
from agent.repair.sb_consistency_estimators.bird_location_consistency import BirdLocationConsistencyEstimator
from agent.repair.sb_consistency_estimators.block_dead_consistency import BlockNotDeadConsistencyEstimator
from agent.repair.sb_consistency_estimators.pig_dead_consistency import PigDeadConsistencyEstimator

logger = logging.getLogger('sb_repair')


class ScienceBirdsConsistencyEstimator(DomainConsistency):
    """
    Checks consistency for SB
    """

    def __init__(self, use_simplified_problems=True):
        # ExternalAgentLocationConsistencyEstimator()
        super().__init__([BirdLocationConsistencyEstimator(), BlockNotDeadConsistencyEstimator(),
                          PigDeadConsistencyEstimator()])
        self.use_simplified_problems = use_simplified_problems

    def _get_traces_from_simulator(self, observation, meta_model, simulator: PddlPlusSimulator, delta_t):
        problem = meta_model.create_pddl_problem(observation.get_initial_state())
        if self.use_simplified_problems:
            problem = meta_model.create_simplified_problem(problem)
        domain = meta_model.create_pddl_domain(observation.get_initial_state())
        domain = PddlPlusGrounder().ground_domain(domain, problem)
        plan = observation.get_pddl_plan(meta_model)
        (_, _, expected_trace,) = simulator.simulate(plan, problem, domain, delta_t=delta_t)
        observed_seq = observation.get_pddl_states_in_trace(meta_model)
        return expected_trace, observed_seq, plan

    def consistency_from_simulator(self, observation, meta_model: ScienceBirdsMetaModel,
                                   simulator: PddlPlusSimulator = NyxPddlPlusSimulator(),
                                   delta_t: float = settings.SB_DELTA_T):
        """
        Computes the consistency of a given observation w.r.t the given meta model using the given simulator
        NOTICE: Using here the simplified problem due to SB domain's complexity
        """
        # TODO remove\rename\remodel this function appropriately. The "simpler model" used here is only one bird at a
        #  time (not all birds at once), so probably has no effect on simulation. Should check and if so remove
        #  simplification.
        try:
            expected_trace, observed_seq, plan = self._get_traces_from_simulator(observation, meta_model, simulator,
                                                                                 delta_t)
            consistency = self.consistency_from_trace(expected_trace, observed_seq, plan, delta_t=delta_t, agg_func=sum)
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
