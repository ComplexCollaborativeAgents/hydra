from agent.consistency.pddl_plus_simulator import *
from agent.consistency.consistency_estimator import *
from agent.planning.pddl_meta_model import *
import heapq

logger = logging.getLogger("meta_model_repair")

PLAN_FAILED_CONSISTENCY_VALUE = 1000 # A constant representing the inconsistency value of a meta model in which the executed plan is inconsistent

''' An abstract class intended to repair a given PDDL+ meta model until it matches the observed behavior '''
class MetaModelRepair(): # TODO: Remove this class

    ''' Repair the given domain and plan such that the given plan's expected outcome matches the observed outcome'''
    def repair(self,
               pddl_meta_model,
               observation, delta_t=1.0):

        simulator = PddlPlusSimulator()
        sb_state = observation.state
        pddl_plan = observation.get_pddl_plan(pddl_meta_model)
        observed_states = observation.get_trace(pddl_meta_model)

        pddl_domain = pddl_meta_model.create_pddl_domain(sb_state)
        pddl_problem = pddl_meta_model.create_pddl_problem(sb_state)
        (_,_,expected_obs) = simulator.simulate(pddl_plan, pddl_problem, pddl_domain, delta_t)

        manipulator_itr = self.choose_manipulator()
        while self.is_consistent(expected_obs, observed_states)==False:
            manipulator = next(manipulator_itr)
            manipulator.apply_change(pddl_meta_model)
            pddl_domain = pddl_meta_model.create_pddl_domain(sb_state)
            pddl_problem = pddl_meta_model.create_pddl_problem(sb_state)
            (_, _, expected_obs) = simulator.simulate(pddl_plan, pddl_problem, pddl_domain, delta_t)

        return pddl_meta_model

    ''' The first parameter is a list of (state, time) pairs, the second is just a list of states. 
     Checks if they can be aligned. '''
    def is_consistent(self, timed_state_seq: list, state_seq: list, delta_t =0.05):
        raise NotImplementedError("Not yet")

    def choose_manipulator(self):
        raise NotImplemented("Not yet")

'''
A greedy best-first search model repair implementation. 
'''
class GreedyBestFirstSearchMetaModelRepair(MetaModelRepair):
    def __init__(self, fluents_to_repair,
                 consistency_estimator,
                 deltas,
                 consistency_threshold=2,
                 max_iteration=30):

        self.consistency_estimator = consistency_estimator
        self.fluents_to_repair = fluents_to_repair
        self.deltas =deltas
        self.consistency_threshold = consistency_threshold
        self.max_iterations = max_iteration
        self.simulator = PddlPlusSimulator()

        # Init other fields
        self.current_delta_t = None
        self.current_meta_model = None

    ''' The heursitic to use to prioritize repairs'''
    def _heuristic(self, repair, consistency):
        change_cardinality = 0
        for i, change in enumerate(repair):
            change_cardinality=change_cardinality+ abs(change/self.deltas[i])
        return consistency+change_cardinality

    ''' Repair the given domain and plan such that the given plan's expected outcome matches the observed outcome'''
    def repair(self,
               pddl_meta_model: MetaModel,
               observation, delta_t=1.0):

        self.current_delta_t = delta_t
        self.current_meta_model = pddl_meta_model

        # Initialize OPEN
        open = []
        repair = [0]* len(self.fluents_to_repair) # Repair is a list, in order of the fluents_to_repair list
        base_consistency = self._compute_consistency(repair, observation)
        priority = self._heuristic(repair, base_consistency)
        heapq.heappush(open, [priority, repair])

        generated_repairs = set() # A set of all generated repaired. This is used for pruning duplicate repairs

        iteration = 0
        # fig = test_utils.plot_observation(observation) # For debug
        incumbent_consistency = base_consistency
        incumbent_repair = repair

        while iteration < self.max_iterations and \
                not (self.is_incumbent_good_enough(incumbent_consistency)
                     and np.any(incumbent_repair)): # Last condition is designed to prevent returning an empty repair
            [_, repair] = heapq.heappop(open)
            new_repairs = self.expand(repair)
            for new_repair in new_repairs:
                new_repair_tuple = tuple(new_repair)
                if new_repair_tuple not in generated_repairs: # If  this is a new repair
                    generated_repairs.add(new_repair_tuple) # To allow duplicate detection
                    consistency = self._compute_consistency(new_repair, observation)
                    priority = self._heuristic(new_repair, consistency)
                    heapq.heappush(open, [priority, new_repair])

                    # If there is a new best solution
                    if consistency < incumbent_consistency:
                        incumbent_consistency = consistency
                        incumbent_repair = new_repair

            iteration = iteration+1

        # If found a useful consistency - apply it to the current meta model
        if base_consistency>incumbent_consistency:
            logging.debug("Found a useful repair! %s,\n consistency gain=%.2f" % (str(incumbent_repair),
                                                                                  base_consistency - incumbent_consistency))
            self._do_change(incumbent_repair)
        return incumbent_repair, incumbent_consistency

    ''' Return True if the incumbent is good enough '''
    def is_incumbent_good_enough(self, consistency: float):
        return consistency < self.consistency_threshold

    ''' Expand the given repair by generating new repairs from it '''
    def expand(self, repair):
        new_repairs = []
        for i, fluent in enumerate(self.fluents_to_repair):
            if repair[i] >= 0:
                new_repair = list(repair)
                new_repair[i] = new_repair[i] + self.deltas[i]
                new_repairs.append(new_repair)
            if repair[i] <= 0:  # Note: if repair has zero for the current fluent, add both +delta and -delta states to open
                new_repair = list(repair)
                new_repair[i] = new_repair[i] - self.deltas[i]
                new_repairs.append(new_repair)
        return new_repairs

    ''' Computes the consistency score for the given delta state'''
    def _compute_consistency(self, repair: dict, observation: ScienceBirdsObservation):
        # Apply change
        self._do_change(repair)

        try:
            expected_trace, plan = self.simulator.get_expected_trace(observation, self.current_meta_model, self.current_delta_t)
            observed_seq = observation.get_trace(self.current_meta_model)
            consistency = self.consistency_estimator.estimate_consistency(expected_trace, observed_seq)
        except InconsistentPlanError: # Sometimes the repair makes the executed plan be inconsistent, e.g., its preconditions are not satisfied
            consistency = PLAN_FAILED_CONSISTENCY_VALUE

        self._undo_change(repair)
        return  consistency

    def _do_change(self, change : dict):
        for i, change_to_fluent in enumerate(change):
            self.current_meta_model.constant_numeric_fluents[self.fluents_to_repair[i]] = \
                self.current_meta_model.constant_numeric_fluents[self.fluents_to_repair[i]] + change_to_fluent

    def _undo_change(self, change: dict):
        for i,change_to_fluent in enumerate(change):
            self.current_meta_model.constant_numeric_fluents[self.fluents_to_repair[i]] = \
                self.current_meta_model.constant_numeric_fluents[self.fluents_to_repair[i]] - change_to_fluent
