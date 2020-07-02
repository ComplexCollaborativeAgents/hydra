import numpy as np
from agent.planning.pddl_meta_model import *
from agent.perception.perception import *
import subprocess

''' A small object that represents an observation of the SB game, containing the values
(state, action, intermedidate_states, reward)'''
class ScienceBirdsObservation:
    def __init__(self):
        self.state = None # An SBState
        self.action = None # An action performed at that state.
        self.intermediate_states = None # The  sequence of intermediates states observed after doing the action
        self.reward = 0 # The reward obtained from performing an action


    ''' Returns a sequence of PDDL states that are the observed intermediate states '''
    def get_trace(self, meta_model: MetaModel = MetaModel()):
        observed_state_seq = []
        perception = Perception()
        for intermediate_state in self.intermediate_states:
            if isinstance(intermediate_state, SBState):
                intermediate_state = perception.process_sb_state(intermediate_state)
            observed_state_seq.append(meta_model.create_pddl_state(intermediate_state))
        return observed_state_seq

    ''' Stores the observation in a file specified by the prefix'''
    def log_observation(self,prefix):
        trace_dir = "{}/agent/consistency/trace/observations".format(settings.ROOT_PATH)
        cmd = "mkdir -p {}".format(trace_dir)
        subprocess.run(cmd, shell=True)
        pickle.dump(self, open("{}/{}_observation.p".format(trace_dir,prefix), 'wb'))

    def load_observation(full_path):
        return pickle.load(open(full_path, 'rb'))

'''
An abstract class for checking if a given sequence of (state, time) pairs can be consistent with a given sequence of states.  
'''
class ConsistencyEstimator:
    ''' The first parameter is a list of (state,time) pairs, the second is just a list of states.
     Returns a positive number  that represents the possible consistency between the sequences,
     where zero means fully consistent. '''
    def estimate_consistency(self, timed_state_seq: list, state_seq: list, delta_t: float = 0.05):
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
    def estimate_consistency(self, simulation_trace: list, state_seq: list, delta_t: float = 0.05):

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
    def __init__(self, fluent_names, unique_prefix_size=7, discount_factor=0.9):
        self.fluent_names = []
        for fluent_name in fluent_names:
            if isinstance(fluent_name,list):
                fluent_name = tuple(fluent_name) # Need a hashable object, to turn it to tuples. TODO: Change all fluent names to tuples
            assert isinstance(fluent_name,tuple)
            self.fluent_names.append(fluent_name)

        self.discount_factor = discount_factor
        self.unique_prefix_size = unique_prefix_size

    ''' The first parameter is a list of (state,time) pairs, the second is just a list of states '''
    ''' Current implementation ignores order, and just looks for the best time for each state in the state_seq, 
    and ignore cases where the fluent is not in the un-timed state seqqaiming to minimize its distance from the fitted piecewise-linear interpolation. 
    Returns a value representing how cons
    '''
    def estimate_consistency(self, simulation_trace: list, state_seq: list, delta_t: float = 0.05):
        t_values, fluent_to_expected_values = self._fit_expected_values(simulation_trace, delta_t=delta_t)

        # Check max error: compute the error for every state w.r.t every time. Return max error found
        max_error = 0
        unique_prefix_counter = 0
        old_obs_state = None
        discount = 1.0
        for state in state_seq:
            if state!=old_obs_state: # Ignore duplicate states
                old_obs_state = state
                unique_prefix_counter = unique_prefix_counter+1
                (best_fit_t, min_error) = self._compute_best_fit(state, t_values, fluent_to_expected_values)

                min_error = min_error *discount # Weight of errors later in the sequence is smaller TODO: Think about this more
                discount = discount*self.discount_factor

                if min_error>max_error:
                    max_error= min_error

                if unique_prefix_counter>=self.unique_prefix_size: # We only consider a limited prefix of the observed sequence of states
                    break

        return max_error

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
    def __init__(self, unique_prefix_size = 7,discount_factor=0.9):
        self.unique_prefix_size=unique_prefix_size
        self.discount_factor = discount_factor

    ''' Estimate consitency by considering the location of the birds in the observed state seq '''
    def estimate_consistency(self, simulation_trace: list, state_seq: list, delta_t: float = 0.05):
        # Get birds
        birds = set()
        for state in state_seq:
            if len(birds)==0: # TODO: Discuss how to make this work. Currently a new bird suddenly appears
                birds = state.get_birds()
            else:
                break

        fluent_names = []
        for bird in birds:
            fluent_names.append(('x_bird', bird))
            fluent_names.append(('y_bird', bird))

        consistency_checker = NumericFluentsConsistencyEstimator(fluent_names, self.unique_prefix_size, self.discount_factor)
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