import os
from agent.perception.novelty_detection.novelty_detector import ObsToState
from agent.perception.novelty_detection.novelty_detector import NoveltyDetector

import pickle
import pdb



if __name__=="__main__":
    model_path = "agent/perception/novelty_detection/params/model.pt"
    class_info_path = "agent/perception/novelty_detection/params/class_info.npz"
    novelty_detector = NoveltyDetector(model_path, class_info_path)

    data_path = 'angry_birds/data/50_level_1_type_10_novelties/random/'
    data_obs = os.listdir(data_path)
    Obs_to_State = ObsToState()

    for obs in data_obs:
        print(obs)
        observation = pickle.load(open(data_path+obs,"rb"))
        state, image = Obs_to_State.Obs_to_StateImage(observation)
        novelty_detector.init_state(state, image)
        novelty_detector.evalaute()
