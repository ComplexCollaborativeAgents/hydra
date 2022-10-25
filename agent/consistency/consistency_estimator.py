import math

import matplotlib
import numpy as np
from typing import List

from agent.consistency.fast_pddl_simulator import *
from agent.consistency.nyx_pddl_simulator import NyxPddlPlusSimulator
from agent.consistency.trace_visualizer import plot_sb_observation, plot_expected_trace_for_obs
from tests import test_utils

# Defaults
DEFAULT_DELTA_T = settings.SB_DELTA_T
DEFAULT_PLOT_OBS_VS_EXP = False
CONSISTENCY_CHECK_FAILED_VALUE = 1000


class AspectConsistency:
    """
    Computes inconsistency estimate for a particular aspect of a domain\problem, e.g. pigsDead or InventoryValues.
    """
    # A constant representing the inconsistency value of a meta model in which the executed plan is inconsistent
    PLAN_FAILED_CONSISTENCY_VALUE = 1000
    default_obs_prefix = 50
    default_discount_factor = 0.9
    default_consistency_threshold = 0.01

    def __init__(self, fluent_names, fluent_template=None, obs_prefix=default_obs_prefix,
                 discount_factor=default_discount_factor,
                 consistency_threshold=default_consistency_threshold):
        """ Specify which fluents to check, and the size of the observed sequence prefix to consider.
        This is because we acknowledge that later in the observations, our model is less accurate. """

        self.fluent_template = fluent_template
        self.fluent_names = []
        for fluent_name in fluent_names:
            if isinstance(fluent_name, list):
                fluent_name = tuple(
                    fluent_name)  # Need a hashable object, to turn it to tuples. TODO: Change all fluent names to tuples
            self.fluent_names.append(fluent_name)

        self.discount_factor = discount_factor
        self.obs_prefix = obs_prefix
        self.consistency_threshold = consistency_threshold

    def consistency_from_trace(self, simulation_trace: list, state_seq: list, delta_t: float = DEFAULT_DELTA_T):
        """
        The first parameter is a list of (state,time) pairs of the expected (simulated) plan.
        the second is a list of observed (actually happened) states.
        Returns a positive number that represents the possible consistency between the sequences,
        where zero means fully consistent.
        """
        raise NotImplementedError()

    def get_fluents(self):
        """
        Returns the fluents this aspect uses to calculate consistency.
        This is a good start on what fluents affect this aspect, but by no means a complete list.
        """
        return self.fluent_names

    def _consistency_from_matched_trace(self, simulation_trace: list, state_seq: list, delta_t: float = DEFAULT_DELTA_T):
        """
        Returns a consistency value for the given traces, assuming the sequences of states are (at least almost) matched
        in time. This is used by cartpole (and ++) and polycraft.
        """
        # if len(simulation_trace) < len(state_seq):
        #     return len(state_seq) - len(simulation_trace)

        # Compute consistency of every observed state
        consistency_per_state = self._compute_consistency_per_matched_state(simulation_trace, state_seq)

        # Aggregate the consistency
        discount = 1.0
        max_error = 0

        for i, consistency in enumerate(consistency_per_state):
            if i > self.obs_prefix:
                break
            if consistency > self.consistency_threshold:
                weighted_error = consistency * discount
                if max_error < weighted_error:
                    max_error = weighted_error

            discount = discount * self.discount_factor

        return max_error

    def _compute_consistency_per_matched_state(self, expected_state_seq: list, observed_states: list):
        """
        Computes inconsistency values for each state, finding the closest time-stamp match for each observation.
        Returns a vector of values, one per observed state, indicating how much it is consistent with the simulation.
        The first parameter is a list of (state,time) pairs, the second is just a list of states
        """

        # If we expected the trade to end before it really did - this is inconsistent
        # assert len(expected_state_seq) >= len(observed_states)

        if len(expected_state_seq) < len(observed_states):
            return [abs(len(observed_states) - len(expected_state_seq))]

        exp_to_obs_step_ratio = 1
        consistency_per_state = []
        for obs_index, obs_state in enumerate(observed_states):
            exp_state = expected_state_seq[int(exp_to_obs_step_ratio * obs_index)][0]
            error = 0
            for fluent_name in self.fluent_names:
                if fluent_name not in obs_state:  # TODO: A design choice. Ignore missing fluent values
                    continue
                if fluent_name not in exp_state:
                    continue

                exp_fluent_value = float(exp_state[fluent_name])
                obs_fluent_value = float(obs_state[fluent_name])
                error = error + (exp_fluent_value - obs_fluent_value) * (exp_fluent_value - obs_fluent_value)
            consistency_per_state.append(math.sqrt(error))
            if obs_index >= self.obs_prefix:  # We only consider a limited prefix of the observed sequence of states
                break
        return consistency_per_state

    def _consistency_from_unmatched_trace(self, simulation_trace: list, state_seq: list, delta_t: float = DEFAULT_DELTA_T):
        """
         Computes an inconsistency value from traces that do not have matching timestamps. This method currently only
         used for Science Birds.
        """
        # Only consider states with some info regarding the relevant fluents
        states_with_info = []
        for state in state_seq:
            has_info = False
            for fluent_name in self.fluent_names:
                if fluent_name in state:
                    has_info = True
                    break
            if has_info:
                states_with_info.append(state)
        state_seq = states_with_info

        # Compute consistency of every observed state
        consistency_per_state = self._compute_consistency_per_unmatched_state(simulation_trace, state_seq, delta_t)

        # Aggregate the consistency
        discount = 1.0
        max_error = 0
        for i, consistency in enumerate(consistency_per_state):
            if i > self.obs_prefix:
                break
            if consistency > self.consistency_threshold:
                weighted_error = consistency * discount
                if max_error < weighted_error:
                    max_error = weighted_error

            discount = discount * self.discount_factor

        return max_error

    def _compute_consistency_per_unmatched_state(self, expected_state_seq: list, observed_states: list,
                                                 delta_t: float = DEFAULT_DELTA_T):
        """
        This method currently only used for Science Birds.
        Computes inconsistency values for observations that do not have matching timestamps as the simulation.
        Current implementation ignores order, and just looks for the best time for each state in the state_seq,
        and ignore cases where the fluent is not in the un-timed state seq aiming to minimize its distance from the
        fitted piecewise-linear interpolation.
        """
        t_values, fluent_to_expected_values = self._linear_interpolate_fluents(expected_state_seq, delta_t=delta_t)
        consistency_per_state = []
        for i, state in enumerate(observed_states):
            (best_fit_t, min_error) = self._compute_best_fit(state, t_values, fluent_to_expected_values)
            consistency_per_state.append(min_error)
            if i >= self.obs_prefix:  # We only consider a limited prefix of the observed sequence of states
                break
        return consistency_per_state

    def _linear_interpolate_fluents(self, simulation_trace, delta_t=0.01):
        """
        This method currently only used for Science Birds.
        Create a piecewise linear interpolation for the given timed_state_seq
        """
        # Get values over time for each fluent
        fluent_to_values = dict()
        fluent_to_times = dict()
        for (state, t, _) in simulation_trace:
            for fluent_name in self.fluent_names:
                if fluent_name in state.numeric_fluents:
                    fluent_value = state[fluent_name]
                    if fluent_name not in fluent_to_times:
                        fluent_to_times[fluent_name] = []
                        fluent_to_values[fluent_name] = []
                    fluent_to_times[fluent_name].append(t)
                    fluent_to_values[fluent_name].append(float(fluent_value))

                    # Fit a piecewise linear function to each fluent
        max_t = max([fluent_to_times[fluent_name][-1] for fluent_name in fluent_to_times])
        fluent_to_expected_values = dict()
        all_t_values = np.arange(0, max_t, delta_t)
        for fluent_name in self.fluent_names:
            t_values = fluent_to_times[fluent_name]
            fluent_values = fluent_to_values[fluent_name]
            fitted_values = np.interp(all_t_values, t_values, fluent_values)
            fluent_to_expected_values[fluent_name] = fitted_values
        return all_t_values, fluent_to_expected_values

    def _compute_best_fit(self, state, t_values, fluent_to_expected_values):
        """
        This method currently only used for Science Birds.
        Compute the t value that best fits the given state. Returns this t and the error at that time, i.e., the
        difference between the state's fluent values and their expected values at the best fit time. """
        best_fit_error = float('inf')
        best_t = -1
        for t in range(len(t_values)):
            error_at_t = 0
            for fluent_name in self.fluent_names:
                if fluent_name not in state:  # TODO: A design choice. Ignore missing fluent values
                    continue
                fluent_value = float(state[fluent_name])
                consistent_fluent_value = fluent_to_expected_values[fluent_name][t]
                fluent_error = abs(fluent_value - consistent_fluent_value)
                if fluent_error > error_at_t:
                    error_at_t = fluent_error
            if error_at_t < best_fit_error:
                best_fit_error = error_at_t
                best_t = t
        return best_t, best_fit_error

    def _objects_in_last_frame(self, simulation_trace: list, observations: list, in_sim=True, in_obs=True):
        """
        This method currently only used for Science Birds.
        Compares the objects matching the given fluent pattern in the last frame.
        Returns the total delta according to flags:
        in_sim: objects that are 'alive' in simulation but 'dead' in observation
        in_obs: objects that are 'dead' in simulation but 'alive' in observation
        Default is both.
        """
        non_matching_obj = 0
        last_state_in_obs = observations[-1]
        obs_objs = last_state_in_obs.get_objects(self.fluent_template)
        last_state_in_sim = simulation_trace[-1][0]
        sim_objs = last_state_in_sim.get_objects(self.fluent_template)
        for obj in sim_objs:
            # All object fluents exist in the simulation, but the object might be 'dead'
            if in_sim and last_state_in_sim[(self.fluent_template + '_life', obj)] > 0 and obj not in obs_objs:
                non_matching_obj += 1
            if in_obs and last_state_in_sim[(self.fluent_template + '_life', obj)] <= 0 and obj in obs_objs:
                non_matching_obj += 1
        return non_matching_obj


    def _filter_trace(self, simulation_trace: list, state_seq: list, filter_func=lambda x: True):
        """
        Filters trajectories according to given function.
        Useful for e.g.
           SB birds with condition (launched && not tapped && not hit anything)
           SB external agents
           Cartpole external agents
        """
        pass  # TODO This method currently unused, but might be useful when we improve the consistency estimation.

    def _trajectory_compare(self, simulation_trace: list, state_seq: list, delta_t: float = DEFAULT_DELTA_T):
        """
        This method currently only used for Science Birds.
        Compares fluents along a trajectory.
        """
        # in future: optionally filtering according to some conditional function.
        # Get objects
        objects = set()
        ob_name = self.fluent_template
        for [state, _, _] in simulation_trace:
            if len(objects) == 0:  # TODO: Are some objects not in the first frame?
                objects = state.get_objects(ob_name)
            else:
                break

        self.fluent_names = []
        for obj in objects:
            self.fluent_names.append(('x_' + ob_name, obj))
            self.fluent_names.append(('y_' + ob_name, obj))

        return self._consistency_from_unmatched_trace(simulation_trace, state_seq, delta_t)


class DomainConsistency:
    """
    Aggregates consistency estimation for a domain, based on a set of aspect-specific values.
    """

    def __init__(self, aspect_estimators: List[AspectConsistency]):
        self.aspect_estimators = aspect_estimators

    def consistency_from_observations(self, meta_model, simulator, observation, delta_t):
        """
        Used to get a consistency value directly from an observation - used often at the moment, but restructuring and
        improving the repair framework should make it obsolete.
        """
        expected_states, observed_states = self.get_traces_from_simulator(observation, meta_model, simulator, delta_t)
        return self.consistency_from_trace(expected_states, observed_states, delta_t)

    def get_traces_from_simulator(self, observation, meta_model, simulator: PddlPlusSimulator, delta_t):
        """ Generates simulation and observation traces for a given observation object """
        expected_trace, plan = simulator.get_expected_trace(observation, meta_model, delta_t)
        observed_seq = observation.get_pddl_states_in_trace(meta_model)
        return expected_trace, observed_seq

    def consistency_from_trace(self, simulation_trace: list, state_seq: list, delta_t: float = DEFAULT_DELTA_T,
                               agg_func=max):
        """
        aggregates inconsistency values for this domain. Default aggregation function is 'max'.
        """
        # example for polycraft: if there are unknown objects, return 1. Otherwise, weighted average of aspects.
        inconsistencies = [c_e.consistency_from_trace(simulation_trace, state_seq, delta_t)
                           for c_e in self.aspect_estimators]
        return agg_func(inconsistencies)

    def suggested_fluents(self):
        """
        Returns fluents that were inconsistent in the last check, and optionally additional fluents that might be
        relevant to repair said inconsistency.
        """
        # presumably by querying the get_fluents() method of each aspect consistency estimator, plus some causal graph logic?
        pass

    def suggested_MMOs(self):
        """
        Model manipulators that are likely relevant to repair. These are manipulators related to the inconsistencies
        found.
        """
        # Following Roni's idea. This might belong in the repair framework, rather than here.
        pass



# TODO: The following code only used in (obsolete) tests, and can be safely removed.
def diff_traces(trace1, trace2: list):
    """
    A utility function for comparing (state, time) sequences.
    It outputs a list of differences between the sequences.
    """
    if len(trace1) != len(trace2):
        return ["len(self)=%d != len(other)=%d" % (len(trace1), len(trace2))]
    diff_list = []
    for i in range(len(trace1)):
        (state1, t1) = trace1[i]
        (state2, t2) = trace2[i]
        if t1 != t2:
            diff_list.append("t=%s, other_t=%d" % (t1, t2))
        else:
            state_diff = diff_pddl_states(state1, state2)
            for diff in state_diff:
                diff_list.append("t=%s, %s" % (t1, diff))
    return diff_list


def diff_pddl_states(state1, state2):
    """
    Computes a diff of two states, and append the list of diffs to the given diff_list.
    """
    diff_list = list()
    for fluent_name in state1.numeric_fluents:
        if fluent_name not in state2.numeric_fluents:
            diff_list.append("%s exists in state1 but not in state2" % str(fluent_name))
        else:
            state_value = state1[fluent_name]
            other_value = state2[fluent_name]
            if state_value != other_value:
                diff_list.append("%s values are different (%s!=%s)" % (
                    str(fluent_name), str(state_value), str(other_value)))
    for fluent_name in state2.numeric_fluents:
        if fluent_name not in state1.numeric_fluents:
            diff_list.append("%s exists in state2 but not in state1" % str(fluent_name))
    return diff_list


def check_obs_consistency(observation,
                          meta_model,
                          consistency_checker: DomainConsistency,
                          simulator: PddlPlusSimulator = NyxPddlPlusSimulator(),
                          plot_obs_vs_exp=DEFAULT_PLOT_OBS_VS_EXP,
                          speedup_factor=1.0):
    """
    Checks if an observation is consistent with a given metamodel
    """
    if plot_obs_vs_exp:
        matplotlib.interactive(True)
        plot_axes = plot_sb_observation(observation)
        plot_expected_trace_for_obs(meta_model, observation, ax=plot_axes)
    try:
        simulation_trace, observed_trace = consistency_checker.get_traces_from_simulator(observation, meta_model,
                                                                                         simulator,
                                                                                         meta_model.delta_t * speedup_factor)
        consistency_value = consistency_checker.consistency_from_trace(simulation_trace, observed_trace,
                                                                       meta_model.delta_t * speedup_factor)
    except ValueError:
        consistency_value = CONSISTENCY_CHECK_FAILED_VALUE
    return consistency_value
