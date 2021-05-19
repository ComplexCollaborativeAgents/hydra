from agent.consistency.consistency_estimator import *


class SequenceConsistencyEstimator(ConsistencyEstimator):
    ''' Checks consistency by trying to align sequences of fluents '''

    def __init__(self, fluent_names, obs_prefix=50, discount_factor=0.9, consistency_threshold = 0.01):
        ''' Specify which fluents to check, and the size of the observed sequence prefix to consider.
        This is because we acknowledge that later in the observations, our model is less accurate. '''

        self.fluent_names = []
        for fluent_name in fluent_names:
            if isinstance(fluent_name,list):
                fluent_name = tuple(fluent_name) # Need a hashable object, to turn it to tuples. TODO: Change all fluent names to tuples
            self.fluent_names.append(fluent_name)

        self.discount_factor = discount_factor
        self.obs_prefix = obs_prefix
        self.consistency_threshold = consistency_threshold

    def estimate_consistency(self, simulation_trace: list, state_seq: list, delta_t: float = DEFAULT_DELTA_T):
        ''' Returns a value indicating the estimated consistency. '''

        if len(simulation_trace)<len(state_seq):
            return len(state_seq)-len(simulation_trace)

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

    def compute_consistency_per_state(self, expected_state_seq: list, observed_states: list, delta_t: float = DEFAULT_DELTA_T):
        ''' Returns a vector of values, one per observed state, indicating how much it is consistent with the simulation.
        The first parameter is a list of (state,time) pairs, the second is just a list of states 
        Current implementation ignores order, and just looks for the best time for each state in the state_seq, 
        and ignore cases where the fluent is not in the un-timed state seqq aiming to minimize its distance from the fitted piecewise-linear interpolation. 
        '''
        
        # If we expected the trade to end before it really did - this is inconsistent
        assert len(expected_state_seq)>=len(observed_states)

        exp_to_obs_step_ratio = 1
        consistency_per_state = []
        for obs_index, obs_state in enumerate(observed_states):
            exp_state = expected_state_seq[int(exp_to_obs_step_ratio*obs_index)][0]
            error = 0
            for fluent_name in self.fluent_names:
                if fluent_name not in obs_state:  # TODO: A design choice. Ignore missing fluent values
                    continue
                if fluent_name not in exp_state:
                    continue

                exp_fluent_value = float(exp_state[fluent_name])
                obs_fluent_value = float(obs_state[fluent_name])
                error = error + (exp_fluent_value-obs_fluent_value)*(exp_fluent_value-obs_fluent_value)
            consistency_per_state.append(math.sqrt(error))
            if obs_index>=self.obs_prefix: # We only consider a limited prefix of the observed sequence of states
                break
        return consistency_per_state