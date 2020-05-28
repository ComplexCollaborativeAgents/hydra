from agent.consistency.pddl_plus_simulator import PddlPlusSimulator
from agent.consistency.consistency_estimator import *
from agent.planning.pddl_meta_model import *
import heapq

logger = logging.getLogger("meta_model_repair")

''' An abstract class intended to repair a given PDDL+ meta model until it matches the observed behavior '''
class MetaModelRepair():

    ''' Repair the given domain and plan such that the given plan's expected outcome matches the observed outcome'''
    def repair(self,
               pddl_meta_model: MetaModel,
               sb_state: SBState,
               pddl_plan : PddlPlusPlan,
               observed_states : list, delta_t = 0.05):
        simulator = PddlPlusSimulator()
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
    def is_consistent(self, timed_state_seq: list, state_seq: list):
        raise NotImplementedError("Not yet")

    def choose_manipulator(self):
        raise NotImplemented("Not yet")

'''
A greedy best-first search model repair implementation. 
'''
class GreedyBestFirstSearchMetaModelRepair(MetaModelRepair):
    def __init__(self, fluents_to_repair, consistency_estimator,
                 deltas,
                 consistency_threshold=30,
                 max_iteration=30):

        self.consistency_estimator = consistency_estimator
        self.fluents_to_repair = fluents_to_repair
        self.deltas =deltas
        self.consistency_threshold = consistency_threshold
        self.max_iterations = max_iteration
        self.simulator = PddlPlusSimulator()

    ''' The heursitic to use to prioritize repairs'''
    def _heuristic(self, repair, consistency):
        change_cardinality = 0
        for i, change in enumerate(repair):
            change_cardinality=change_cardinality+ abs(change/self.deltas[i])
        return consistency+change_cardinality

    ''' Repair the given domain and plan such that the given plan's expected outcome matches the observed outcome'''
    def repair(self,
               pddl_meta_model: MetaModel,
               sb_state: SBState,
               pddl_plan: PddlPlusPlan,
               observed_states: list, delta_t=1.0):

        self.current_sb_state = sb_state
        self.current_delta_t = delta_t
        self.current_plan = pddl_plan
        self.current_meta_model = pddl_meta_model

        repair = [0]* len(self.fluents_to_repair) # Repair is a list, in order of the fluents_to_repair list
        base_consistency = self._compute_consistency(repair, observed_states)
        open = []
        priority = self._heuristic(repair, base_consistency)
        heapq.heappush(open, [priority, repair])

        iteration = 0
        incumbent_consistency = base_consistency
        incumbent_repair = repair

        while incumbent_consistency > self.consistency_threshold\
                and iteration<self.max_iterations:
            [_, repair] = heapq.heappop(open)

            # Expand repair
            new_repairs = []
            for i,fluent in enumerate(self.fluents_to_repair):
                if repair[i]>=0:
                    new_repair = list(repair)
                    new_repair[i] = new_repair[i] + self.deltas[i]
                    new_repairs.append(new_repair)
                if repair[i]<=0: # Note: if repair has zero for the current fluent, add both +delta and -delta states to open
                    new_repair = list(repair)
                    new_repair[i] = new_repair[i]-self.deltas[i]
                    new_repairs.append(new_repair)
            for new_repair in new_repairs:
                consistency = self._compute_consistency(new_repair, observed_states)
                priority = self._heuristic(new_repair, consistency)
                heapq.heappush(open, [priority, new_repair])

                # If there is a new best solution
                if consistency<incumbent_consistency:
                    incumbent_consistency = consistency
                    incumbent_repair = new_repair

            iteration = iteration+1

        # If found a useful consistency
        if base_consistency>incumbent_consistency:
            logger.debug("Found a useful repair! %s,\n consistency gain=%.2f" % (str(incumbent_repair),
                                                                                 base_consistency-incumbent_consistency))
            self._do_change(incumbent_repair)
        return pddl_meta_model

    ''' Computes the consistency score for the given delta state'''
    def _compute_consistency(self, delta_state: dict,observed_states: list):
        # Apply change
        self._do_change(delta_state)
        pddl_domain = self.current_meta_model.create_pddl_domain(self.current_sb_state)
        pddl_problem = self.current_meta_model.create_pddl_problem(self.current_sb_state)
        (_, _, timed_state_seq) = self.simulator.simulate(self.current_plan, pddl_problem, pddl_domain, self.current_delta_t)
        self._undo_change(delta_state)
        return self.consistency_estimator.estimate_consistency(timed_state_seq, observed_states)

    def _do_change(self, change : dict):
        for i, change_to_fluent in enumerate(change):
            self.current_meta_model.constant_numeric_fluents[self.fluents_to_repair[i]] = \
                self.current_meta_model.constant_numeric_fluents[self.fluents_to_repair[i]] + change_to_fluent

    def _undo_change(self, change: dict):
        for i,change_to_fluent in enumerate(change):
            self.current_meta_model.constant_numeric_fluents[self.fluents_to_repair[i]] = \
                self.current_meta_model.constant_numeric_fluents[self.fluents_to_repair[i]] - change_to_fluent


    ''' The first parameter is a list of (state, time) pairs, the second is just a list of states. 
     Checks if they can be aligned. '''
    def is_consistent(self, timed_state_seq: list, state_seq: list):
        consistency_value = self.consistency_estimator.estimate_consistency(timed_state_seq, state_seq)
        if consistency_value < self.consistency_threshold:
            return True
        else:
            return False