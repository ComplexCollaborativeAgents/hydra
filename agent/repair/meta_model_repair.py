from abc import ABC, abstractmethod

from agent.consistency.consistency_estimator import *
from agent.planning.sb_meta_model import *
import heapq
import time

logger = logging.getLogger("meta_model_repair")

PLAN_FAILED_CONSISTENCY_VALUE = 1000  # A constant representing the inconsistency value of a meta model in which the executed plan is inconsistent


class RepairModule(ABC):
    """ An abstract class intended to repair a given PDDL+ meta model until it matches the observed behavior.
    Handles all model repair operations, from the moment the agent decided repair is needed.
     - decide what aspect to repair
     - decide if repair is good enough
     - make sure metamodel and domains are updated and repaired correctly.
    """
    def __init__(self, meta_model: MetaModel, consistency_estimator: DomainConsistency):
        self.consistency_estimator = consistency_estimator
        self.meta_model = meta_model

    @abstractmethod
    def repair(self, observations: HydraEpisodeLog, delta_t=0.1):
        # TODO optionally, receive the inconsistency information that instigated this repair call (for efficiency)
        # steps:
        #  1. decide what aspect needs to be repaired
        #  2. choose\create\instantiate appropriate aspect class
        #  3. call repair
        raise NotImplementedError()


class AspectRepair:
    """
    Repairs a specific aspect of a PDDL+ model, e.g. hidden constants, action effects, or goal conditions.
    """

    def __init__(self, meta_model: MetaModel, consistency_estimator: DomainConsistency,
                 simulator: PddlPlusSimulator = NyxPddlPlusSimulator()):
        self.meta_model = meta_model
        self.consistency_estimator = consistency_estimator
        self.simulator = simulator

    def repair(self, observation, delta_t=1.0):
        """ Repair the domain and plan such that the given plan's expected outcome matches the observed outcome"""
        raise NotImplementedError()

    def is_consistent(self, timed_state_seq: list, state_seq: list, delta_t=0.05):
        """ The first parameter is a list of (state, time) pairs, the second is just a list of states.
             Checks if they can be aligned. """
        # raise NotImplementedError()
        pass

    def choose_manipulator(self):
        # raise NotImplemented()
        pass


class GreedyBestFirstSearchConstantFluentMetaModelRepair(AspectRepair):
    """
    A greedy best-first search model repair implementation. Only repairs constant fluents.
    """

    def __init__(self, meta_model, consistency_estimator, fluents_to_repair, deltas, consistency_threshold=2,
                 max_iterations=100, time_limit=180):
        self.fluents_to_repair = fluents_to_repair
        self.deltas = deltas
        self.consistency_threshold = consistency_threshold
        self.max_iterations = max_iterations
        self.time_limit = time_limit

        super().__init__(meta_model, consistency_estimator)

    def _heuristic(self, repair, consistency):
        """ The heuristic to use to prioritize repairs"""
        change_cardinality = 0
        for i, change in enumerate(repair):
            change_cardinality = change_cardinality + abs(change / self.deltas[i])
        return consistency + change_cardinality

    def repair(self, observation, delta_t=1.0):
        """ Repair the given domain and plan such that the given plan's expected outcome matches the observed outcome"""

        start_time = time.time()
        # Initialize OPEN
        open_list = []
        repair = [0] * len(self.fluents_to_repair)  # Repair is a list, in order of the fluents_to_repair list
        expected_states, observed_states = self.consistency_estimator.get_traces_from_simulator(observation,
                                                                                                self.meta_model,
                                                                                                self.simulator, delta_t)
        base_consistency = self.consistency_estimator.consistency_from_trace(expected_states, observed_states, delta_t)
        priority = self._heuristic(repair, base_consistency)
        heapq.heappush(open_list, [priority, repair])

        generated_repairs = set()  # A set of all generated repaired. This is used for pruning duplicate repairs

        iteration = 0
        # fig = test_utils.plot_observation(observation)  # For debug
        incumbent_consistency = base_consistency
        incumbent_repair = repair
        timeout = False

        # print(f'initial consistency: {incumbent_consistency}')

        while iteration < self.max_iterations and not timeout \
                and not (self.is_incumbent_good_enough(incumbent_consistency)
                         and np.any(
                    incumbent_repair) and open_list):  # Last condition is designed to prevent returning an empty repair
            [_, repair] = heapq.heappop(open_list)
            new_repairs = self.expand(repair)
            for new_repair in new_repairs:

                # Check if reached the timeout
                if time.time() - start_time > self.time_limit:
                    timeout = True
                    break
                new_repair_tuple = tuple(new_repair)
                if new_repair_tuple not in generated_repairs:  # If  this is a new repair
                    generated_repairs.add(new_repair_tuple)  # To allow duplicate detection
                    self._do_change(new_repair)
                    try:
                        expected_states, _ = self.consistency_estimator.get_traces_from_simulator(observation,
                                                                                                  self.meta_model,
                                                                                                  self.simulator,
                                                                                                  delta_t)
                        consistency = self.consistency_estimator.consistency_from_trace(expected_states,
                                                                                        observed_states, delta_t)
                    except Exception as ee:
                        # print("error: likely repair ({}) caused a malformed PDDL problem.".format(str(new_repair)))
                        # print(ee)
                        consistency = 1000

                    self._undo_change(new_repair)
                    # print(new_repair, consistency)
                    priority = self._heuristic(new_repair, consistency)
                    heapq.heappush(open_list, [priority, new_repair])

                    # If there is a new best solution
                    if consistency < incumbent_consistency:
                        incumbent_consistency = consistency
                        incumbent_repair = new_repair

            iteration = iteration + 1

        # If found a useful consistency - apply it to the current metamodel
        if base_consistency > incumbent_consistency:
            logging.debug("Found a useful repair! %s,\n consistency gain=%.2f" % (str(incumbent_repair),
                                                                                  base_consistency - incumbent_consistency))
            self._do_change(incumbent_repair)
        return incumbent_repair, incumbent_consistency

    def expand(self, repair):
        """ Expand the given repair by generating new repairs from it """
        new_repairs = []
        for i, fluent in enumerate(self.fluents_to_repair):
            if repair[i] >= 0:
                change_to_fluent = repair[i] + self.deltas[i]
                # if self.current_meta_model.constant_numeric_fluents[self.fluents_to_repair[i]] + change_to_fluent >= 0:
                    # Don't allow negative fluents TODO: Better would be "don't allow fluent to change sign"
                new_repair = list(repair)
                new_repair[i] = change_to_fluent
                new_repairs.append(new_repair)
            if repair[i] <= 0:
                # Note: if repair has zero for the current fluent, add both +delta and -delta states to open
                change_to_fluent = repair[i] - self.deltas[i]
                # if self.current_meta_model.constant_numeric_fluents[self.fluents_to_repair[i]] + change_to_fluent >= 0:
                    # Don't allow negative fluents TODO: Better would be "don't allow fluent to change sign"
                new_repair = list(repair)
                new_repair[i] = change_to_fluent
                new_repairs.append(new_repair)
        return new_repairs

    def _do_change(self, change: list):
        for i, change_to_fluent in enumerate(change):
            self.meta_model.constant_numeric_fluents[self.fluents_to_repair[i]] = \
                self.meta_model.constant_numeric_fluents[self.fluents_to_repair[i]] + change_to_fluent

    def _undo_change(self, change: list):
        for i, change_to_fluent in enumerate(change):
            self.meta_model.constant_numeric_fluents[self.fluents_to_repair[i]] = \
                self.meta_model.constant_numeric_fluents[self.fluents_to_repair[i]] - change_to_fluent

    def is_incumbent_good_enough(self, consistency: float):
        """ Return True if the incumbent is good enough """
        return consistency < self.consistency_threshold
