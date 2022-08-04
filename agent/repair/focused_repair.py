from agent.repair.meta_model_repair import *

DEFAULT_CONSTANT_UPPER_BOUND = 1000
DEFAULT_CONSTANT_LOWER_BOUND = 0

# ONLY USED IN TESTS
# class ModelManipulationOperator:
#     """ A model manipulation operator, used to repair the models """
#     def is_applicable(self, repair):
#         raise NotImplemented()
#
#     def apply(self, repair):
#         raise NotImplemented()


# class ConstantChangeMMO(ModelManipulationOperator):
#     """ An MMO that changes a single fluent """
#
#     def __init__(self, fluent_index, delta, upper_bound=DEFAULT_CONSTANT_UPPER_BOUND,
#                  lower_bound=DEFAULT_CONSTANT_LOWER_BOUND):
#         self.fluent_index = fluent_index
#         self.delta = delta
#         self.upper_bound = upper_bound
#         self.lower_bound = lower_bound
#
#     def is_applicable(self, repair):
#         if self.lower_bound < repair[self.fluent_index] + self.delta < self.upper_bound:
#             return True
#         else:
#             return False
#
#     def apply(self, repair):
#         new_repair = list(repair)
#         new_repair[self.fluent_index] += self.delta
#         return new_repair
#

# class MmoBasedMetaModelRepair(SimulationBasedMetaModelRepair):
#     """ Repair class based on a given set of MMOs. """
#
#     def __init__(self, fluents_to_repair,
#                  consistency_estimator,
#                  deltas,
#                  consistency_threshold=2,
#                  max_iterations=30,
#                  time_limit=1000):
#         super().__init__(fluents_to_repair,
#                          consistency_estimator,
#                          deltas,
#                          consistency_threshold,
#                          max_iterations,
#                          time_limit)
#
#         self.incumbet_repair = None
#         self.incumbet_consistency = 100000
#
#
#
#     def repair(self,
#                pddl_meta_model: ScienceBirdsMetaModel,
#                observation, delta_t=1.0):
#         """ Repair the given domain and plan such that the given plan's expected outcome matches the observed outcome"""
#
#         self._setup_repair(pddl_meta_model, observation, delta_t)
#         base_consistency = self.incumbent_consistency  # Save the base consistency to measure success
#         iteration = 0
#         while iteration < self.max_iterations and len(self.open) > 0 and \
#                 time.time() - self.start_time <= self.time_limit and \
#                 not (self.is_incumbent_good_enough(self.incumbent_consistency) and
#                      np.any(self.incumbent_repair)):  # Last condition is designed to prevent returning an empty repair
#             [_, repair] = heapq.heappop(self.open)
#
#             self.expand(repair)
#
#             iteration = iteration + 1
#
#         # If found a useful consistency - apply it to the current meta model
#         if base_consistency > self.incumbent_consistency:
#             logging.debug("Found a useful repair! %s,\n consistency gain=%.2f" % (str(self.incumbent_repair),
#                                                                                   base_consistency - self.incumbent_consistency))
#             self._do_change(self.incumbent_repair)
#         return self.incumbent_repair, self.incumbent_consistency
#
#     def _setup_repair(self, pddl_meta_model, observation, delta_t):
#         """ Setup the internal parameters before running the repair algorithm """
#         self.start_time = time.time()
#         self.current_delta_t = delta_t
#         self.current_meta_model = pddl_meta_model
#         self.observation = observation
#         # Initialize OPEN
#         self.open = []
#         repair = [0] * len(self.fluents_to_repair)  # Repair is a list, in order of the fluents_to_repair list
#         base_consistency = self.compute_consistency(repair, self.observation)
#         priority = self._heuristic(repair, base_consistency)
#         heapq.heappush(self.open, [priority, repair])
#         self.generated_repairs = set()  # A set of all generated repaired. This is used for pruning duplicate repairs
#
#         # fig = test_utils.plot_observation(observation) # For debug
#         self.incumbent_consistency = base_consistency
#         self.incumbent_repair = repair
#
#         # Initialize mmo_list
#         self.mmo_list = list()
#         MAX_STEPS = 10  # The number of applications of the same MMO we allow
#         for i, fluent in enumerate(self.fluents_to_repair):
#             initial_fluent_value = self.current_meta_model.constant_numeric_fluents[self.fluents_to_repair[i]]
#             self.mmo_list.append(ConstantChangeMMO(i, 1.0 * self.deltas[i], upper_bound=MAX_STEPS * self.deltas[i]))
#             self.mmo_list.append(
#                 ConstantChangeMMO(i, -1.0 * self.deltas[i], lower_bound=-initial_fluent_value / self.deltas[i]))
#
#         return base_consistency
#
#     def expand(self, repair):
#         """ Expand the given repair by generating new repairs from it """
#         for mmo in self.mmo_list:
#             if mmo.is_applicable(repair):
#                 new_repair = mmo.apply(repair)
#                 new_repair_tuple = tuple(new_repair)
#                 if new_repair_tuple not in self.generated_repairs:
#                     self.generated_repairs.add(new_repair_tuple)  # To allow duplicate detection
#                     consistency = self.compute_consistency(new_repair, self.observation)
#                     priority = self._heuristic(new_repair, consistency)
#                     heapq.heappush(self.open, [priority, new_repair])
#
#                     # If there is a new best solution
#                     if consistency < self.incumbent_consistency:
#                         self.incumbent_consistency = consistency
#                         self.incumbent_repair = new_repair
#
#             # Check if reached the timeout
#             if time.time() - self.start_time > self.time_limit:
#                 break
#
#     """ The heursitic to use to prioritize repairs"""
#
#     def _heuristic(self, repair, consistency):
#         change_cardinality = 0
#         for i, change in enumerate(repair):
#             change_cardinality = change_cardinality + abs(change / self.deltas[i])
#         return consistency + change_cardinality

# class FocusedMetaModelRepair(MmoBasedMetaModelRepair):
#     """ A MetaModelRepair"""
#
#     def __init__(self, fluents_to_repair,
#                  consistency_estimator,
#                  deltas,
#                  consistency_threshold=2,
#                  max_iterations=30,
#                  time_limit=1000):
#
#         super().__init__(fluents_to_repair,
#                          consistency_estimator,
#                          deltas,
#                          consistency_threshold=consistency_threshold,
#                          max_iterations=max_iterations,
#                          time_limit=time_limit)
#
#     def _setup_repair(self, pddl_meta_model, observation, delta_t):
#         """ Update the list of MMOs"""
#         super()._setup_repair(pddl_meta_model, observation, delta_t)
#         self._update_mmo_list(self.incumbent_repair, self.incumbent_consistency)
#
#     def _update_mmo_list(self, repair: list, base_consistency: float):
#         """ A heuristic for choosing which MMO to use. With this heuristic, we only ddd to the MMO list only MMOs that impact the consistency. """
#         original_mmo_list = list(self.mmo_list)
#
#         extreme_repair_factor = 3  # Apply the same repair several times to see if it has an impact
#         self.mmo_list.clear()
#         for mmo in original_mmo_list:
#             new_repair = list(repair)
#             for i in range(extreme_repair_factor):
#                 if mmo.is_applicable(new_repair):
#                     new_repair = mmo.apply(new_repair)
#                 else:
#                     break
#
#             # Ideally, compare the trace. TODO: For now, we compare the consistency, assuming that some small difference will always be there
#             consistency = self.compute_consistency(new_repair, self.observation)
#             if base_consistency != consistency:  # This is a heuristic
#                 self.mmo_list.append(mmo)
