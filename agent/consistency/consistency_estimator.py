import matplotlib
from utils.point2D import Point2D
from agent.consistency import pddl_plus_simulator as simulator
from agent.consistency.observation import ScienceBirdsObservation
from agent.perception.perception import *
from agent.planning.pddl_plus import *
from agent.planning.pddl_meta_model import MetaModel
from tests import test_utils

# Defaults
DEFAULT_DELTA_T = 0.02
DEFAULT_PLOT_OBS_VS_EXP = False

'''
An abstract class for checking if a given sequence of (state, time) pairs can be consistent with a given sequence of states.  
'''
class ConsistencyEstimator:
    ''' The first parameter is a list of (state,time) pairs, the second is just a list of states.
     Returns a positive number  that represents the possible consistency between the sequences,
     where zero means fully consistent. '''
    def estimate_consistency(self, timed_state_seq: list, state_seq: list, delta_t: float = DEFAULT_DELTA_T):
        raise NotImplementedError()

''' Checks consistency by considering the value of a single numeric fluent '''
class SingleNumericFluentConsistencyEstimator(ConsistencyEstimator):
    def __init__(self, fluent_name):
        if isinstance(fluent_name,list):
            self.fluent_name = tuple(fluent_name)
        else:
            self.fluent_name = fluent_name

    ''' The first parameter is a list of (state,time) pairs, the second is just a list of states '''
    ''' Current implementation ignores order, and just looks for the best time for each state in the state_seq, 
    aiming to minimize its distance from the fitted piecewise-linear interpolation. 
    Returns a value representing how cons
    '''
    def estimate_consistency(self, simulation_trace: list, state_seq: list, delta_t: float = DEFAULT_DELTA_T):

        # Fit a piecewise linear function based on the timed state sequence
        t_values = list()
        fluent_values = list()
        for (state, t,_) in simulation_trace:
            value = float(state[self.fluent_name])
            t_values.append(t)
            fluent_values.append(value)

        all_t_values = np.arange(0,t_values[-1],delta_t)
        consistent_fluent_values = np.interp(all_t_values, t_values, fluent_values)

        assert len(consistent_fluent_values==len(all_t_values))

        max_error = 0
        for state in state_seq:
            fluent_value = float(state[self.fluent_name])
            error = min([abs(fluent_value-consistent_fluent_values[i])
                          for i in range(len(all_t_values))])
            if max_error<error:
                max_error = error

        return max_error


''' Checks consistency by considering the value of a set of numeric fluents '''
class NumericFluentsConsistencyEstimator(ConsistencyEstimator):

    ''' Specify which fluents to check, and the size of the observed sequence prefix to consider.
    This is because we acknowledge that later in the observations, our model is less accurate. '''
    def __init__(self, fluent_names, obs_prefix=100, discount_factor=0.25, consistency_threshold = 20):
        self.fluent_names = []
        for fluent_name in fluent_names:
            if isinstance(fluent_name,list):
                fluent_name = tuple(fluent_name) # Need a hashable object, to turn it to tuples. TODO: Change all fluent names to tuples
            assert isinstance(fluent_name,tuple)
            self.fluent_names.append(fluent_name)

        self.discount_factor = discount_factor
        self.obs_prefix = obs_prefix
        self.consistency_threshold = consistency_threshold

    ''' Returns a value indicating the estimated consistency. '''
    def estimate_consistency(self, simulation_trace: list, state_seq: list, delta_t: float = DEFAULT_DELTA_T):
        # Only consider states with some info regarding the relevant fluents
        states_with_info = []
        for state in state_seq:
            has_info=False
            for fluent_name in self.fluent_names:
                if fluent_name in state:
                    has_info=True
                    break
            if has_info:
                states_with_info.append(state)
        state_seq = states_with_info

        # Compute consistency of every observed state
        consistency_per_state = self.compute_consistency_per_state(simulation_trace, state_seq, delta_t)

        # Aggregate the consistency
        discount = 1.0
        max_error = 0
        for i, consistency in enumerate(consistency_per_state):
            if i>self.obs_prefix:
                break
            if consistency>self.consistency_threshold:
                weighted_error = consistency*discount
                if max_error < weighted_error:
                    max_error = weighted_error

            discount = discount*self.discount_factor

        return max_error

    ''' Returns a vector of values, one per observed state, indicating how much it is consistent with the simulation.
    The first parameter is a list of (state,time) pairs, the second is just a list of states 
    Current implementation ignores order, and just looks for the best time for each state in the state_seq, 
    and ignore cases where the fluent is not in the un-timed state seqq aiming to minimize its distance from the fitted piecewise-linear interpolation. 
    '''
    def compute_consistency_per_state(self, expected_state_seq: list, observed_states: list, delta_t: float = DEFAULT_DELTA_T):
        t_values, fluent_to_expected_values = self._fit_expected_values(expected_state_seq, delta_t=delta_t)
        consistency_per_state = []
        for i, state in enumerate(observed_states):
            (best_fit_t, min_error) = self._compute_best_fit(state, t_values, fluent_to_expected_values)
            consistency_per_state.append(min_error)
            if i>=self.obs_prefix: # We only consider a limited prefix of the observed sequence of states
                break
        return consistency_per_state

    ''' Create a piecewise linear interpolation for the given timed_state_seq'''
    def _fit_expected_values(self, simulation_trace, delta_t = 0.01):
        # Get values over time for each fluent
        fluent_to_values = dict()
        fluent_to_times = dict()
        for (state, t,_) in simulation_trace:
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
        for fluent_name in self.fluent_names:
            t_values = fluent_to_times[fluent_name]
            fluent_values = fluent_to_values[fluent_name]
            all_t_values = np.arange(0, max_t, delta_t)
            fitted_values = np.interp(all_t_values, t_values, fluent_values)
            fluent_to_expected_values[fluent_name] = fitted_values
        return all_t_values, fluent_to_expected_values

    ''' Compute the t value that best fits the given state. Returns
    this t and the error at that time, i.e., the difference between 
    the state's fluent values and their expected values at the best fit time. '''
    def _compute_best_fit(self, state, t_values, fluent_to_expected_values):
        best_fit_error = float('inf')
        for t in range(len(t_values)):
            error_at_t = 0
            for fluent_name in self.fluent_names:
                if fluent_name not in state: # TODO: A design choice. Ignore missing fluent values
                    continue
                fluent_value = float(state[fluent_name])
                consistent_fluent_value = fluent_to_expected_values[fluent_name][t]
                fluent_error = abs(fluent_value - consistent_fluent_value)
                if fluent_error > error_at_t:
                    error_at_t = fluent_error
            if error_at_t < best_fit_error:
                best_fit_error = error_at_t
                best_t = t
        return (best_t, best_fit_error)


''' Checks consistency by considering the location of the birds '''
class BirdLocationConsistencyEstimator():
    def __init__(self, unique_prefix_size = 100,discount_factor=0.9, consistency_threshold = 20):
        self.unique_prefix_size=unique_prefix_size
        self.discount_factor = discount_factor
        self.consistency_threshold = consistency_threshold

    ''' Estimate consitency by considering the location of the birds in the observed state seq '''
    def estimate_consistency(self, simulation_trace: list, state_seq: list, delta_t: float = DEFAULT_DELTA_T):
        # Get birds
        birds = set()
        for [state, _,_] in simulation_trace:
            if len(birds)==0: # TODO: Discuss how to make this work. Currently a new bird suddenly appears
                birds = state.get_birds()
            else:
                break

        fluent_names = []
        for bird in birds:
            fluent_names.append(('x_bird', bird))
            fluent_names.append(('y_bird', bird))

        consistency_checker = NumericFluentsConsistencyEstimator(fluent_names, self.unique_prefix_size,
                                                                 self.discount_factor,
                                                                 consistency_threshold = self.consistency_threshold)
        return consistency_checker.estimate_consistency(simulation_trace, state_seq, delta_t)


''' Checks consistency by considering the location of the birds '''
class CartpoleConsistencyEstimator():
    def __init__(self, unique_prefix_size = 100,discount_factor=0.9, consistency_threshold = 20):
        self.unique_prefix_size=unique_prefix_size
        self.discount_factor = discount_factor
        self.consistency_threshold = consistency_threshold

    ''' Estimate consitency by considering the location of the birds in the observed state seq '''
    def estimate_consistency(self, simulation_trace: list, state_seq: list, delta_t: float = DEFAULT_DELTA_T):

        fluent_names = []
        fluent_names.append(('x',))
        fluent_names.append(('x_dot',))
        fluent_names.append(('theta',))
        fluent_names.append(('theta_dot',))

        consistency_checker = NumericFluentsConsistencyEstimator(fluent_names, self.unique_prefix_size,
                                                                 self.discount_factor,
                                                                 consistency_threshold = self.consistency_threshold)
        return consistency_checker.estimate_consistency(simulation_trace, state_seq, delta_t)




''' A utility function for comparing (state, time) sequences. 
It outputs a list of differences between the sequences. '''
def diff_traces(trace1, trace2: list):
    if len(trace1)!=len(trace2):
        return ["len(self)=%d != len(other)=%d" % (len(trace1), len(trace2))]
    diff_list= []
    for i in range(len(trace1)):
        (state1, t1) = trace1[i]
        (state2, t2) = trace2[i]
        if t1!=t2:
            diff_list.append("t=%s, other_t=%d" % (t1, t2))
        else:
            state_diff = diff_pddl_states(state1, state2)
            for diff in state_diff:
                diff_list.append("t=%s, %s" % (t1, diff))
    return diff_list


'''
Computes a diff of two states, and append the list of diffs to the given diff_list. 
'''
def diff_pddl_states(state1, state2):
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

''' Checks if an observation is consisten with a given meta model'''
def check_obs_consistency(observation,
                          meta_model,
                          consistency_checker,
                          delta_t : float = DEFAULT_DELTA_T,
                          plot_obs_vs_exp = DEFAULT_PLOT_OBS_VS_EXP):
    if plot_obs_vs_exp:
        matplotlib.interactive(True)
        plot_axes = test_utils.plot_observation(observation)
        test_utils.plot_expected_trace_for_obs(meta_model, observation, ax=plot_axes)

    expected_trace, plan = simulator.get_expected_trace(observation, meta_model, delta_t=delta_t)
    observed_seq = observation.get_trace(meta_model)
    return consistency_checker.estimate_consistency(expected_trace, observed_seq, delta_t=delta_t)