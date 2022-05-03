from agent.consistency.observation import ScienceBirdsObservation
from agent.reward_estimation.nn_utils.obs_to_imgs import *
from agent.reward_estimation.nn_utils.ab_dataset_tensor import *
import pickle
import settings

def extract_state_action_from_observation(observation: ScienceBirdsObservation):
    obsimg = SBObs_to_Imgs()
    state, action, inter_states = obsimg.Obs_to_StateActionNextState(observation)
    print(state)
    reward = observation.reward
    state_img = obsimg.state_to_nD_img(state)
    state_img = np.reshape(state_img, [1, 480, 840, len(obsimg.TYPES)])
    state_img = state_img[:, 100:400, 0:600, :]  #####################################################
    action = np.reshape(action, [1, 5])
    reward = np.reshape(reward, [1, 1])
    # input size for torch.nn.Conv2d : (N, C, W, H)
    state_img = np.transpose(state_img, (0, 3, 1, 2))
    # print(np.shape(state_img), np.shape(action), np.shape(reward)) #(1, 13, 480, 840) (1, 5) (1, 1)
    return state_img, action, reward

def convert_pickle_to_image(path_to_pickle):
    observation = pickle.load(open(path_to_pickle, "rb"))
    obsimg = SBObs_to_Imgs()
    state, action, inter_states = obsimg.Obs_to_StateActionNextState(observation)
    print(state)
    reward = observation.reward
    state_img = obsimg.state_to_nD_img(state)
    state_img = np.reshape(state_img, [1, 480, 840, 13])
    state_img = state_img[:, 100:400, 0:600, :]  #####################################################
    action = np.reshape(action, [1, 5])
    reward = np.reshape(reward, [1, 1])
    # input size for torch.nn.Conv2d : (N, C, W, H)
    state_img = np.transpose(state_img, (0, 3, 1, 2))
    # print(np.shape(state_img), np.shape(action), np.shape(reward)) #(1, 13, 480, 840) (1, 5) (1, 1)
    return state_img, action, reward


def normalization_reward(reward_output):
    reward_max = [4216.0]  # [100000.0]
    reward_min = [0.0]

    np.save("reward_max.npy", np.asarray(reward_max))
    np.save("reward_min.npy", np.asarray(reward_min))
    print("Normalization start")

    reward_output = (reward_output[:, 0] - reward_min[0]) / (reward_max[0] - reward_min[0])
    # with open("state_input_1_1.pkl",'wb') as f:
    #    pickle.dump( state_input, f)
    # with open("action_input_1_1.pkl",'wb') as f:
    #    pickle.dump( action_input, f)
    # with open("reward_output_1_1.pkl",'wb') as f:
    #    pickle.dump( reward_output, f)

    # np.savez("nomarlized_level0_2.npz", state = state_input, action = action_input, reward = reward_output)
    return reward_output


def normalization(state_input, action_input, reward_output):
    state_max = np.load("{}/agent/reward_estimation/state_max.npy".format(settings.ROOT_PATH))
    state_min = np.load("{}/agent/reward_estimation/state_min.npy".format(settings.ROOT_PATH))
    action_max = np.load("{}/agent/reward_estimation/action_max.npy".format(settings.ROOT_PATH))
    action_min = np.load("{}/agent/reward_estimation/action_min.npy".format(settings.ROOT_PATH))
    reward_max = np.load("{}/agent/reward_estimation/reward_max.npy".format(settings.ROOT_PATH))
    reward_min = np.load("{}/agent/reward_estimation/reward_min.npy".format(settings.ROOT_PATH))
    for i in range(13):
        if state_max[i] != state_min[i]:
            state_input[:, i, :, :] = (state_input[:, i, :, :] - state_min[i]) / (state_max[i] - state_min[i])
        else:
            state_input[:, i, :, :] = 0.0

    for j in range(5):
        if action_max[j] != action_min[j]:
            action_input[:, j] = (action_input[:, j] - action_min[j]) / (action_max[j] - action_min[j])
        else:
            action_input[:, j] = 0.0
    reward_output = (reward_output[:, 0] - reward_min[0]) / (reward_max[0] - reward_min[0])
    return state_input, action_input, reward_output