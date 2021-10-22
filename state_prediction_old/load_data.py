import numpy as np
import pickle
import os
import pdb
import cv2


def unpack_obs(obs):
    state = obs['state']
    action = obs['action']
    next_states = obs['next_states']

    return state, action, next_states


if __name__=="__main__":

    num_obs = 10

    state_array = np.empty((480, 840, 13, num_obs), dtype=np.float32)
    next_state_array = np.empty((480, 840, 13, num_obs), dtype=np.float32)
    action_array = np.empty((5, num_obs), dtype=np.float32)

    for i in range(num_obs):
        obs = np.load(os.path.join('obs_data_small', 'obs_data_{}.npz'.format(i + 1)), allow_pickle=True)['obs_list'][()]
        state, action, next_states = unpack_obs(obs)

        state_array[:, :, :, i] = state
        action_array[:, i] = action
        next_state_array[:, :, :, i] = next_states[0]

    print(state_array.shape, state_array.dtype)
    print(action_array.shape, action_array.dtype)
    print(next_state_array.shape, next_state_array.dtype)
    
