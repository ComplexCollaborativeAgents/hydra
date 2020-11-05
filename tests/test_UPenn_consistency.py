import os

from state_prediction.anomaly_detector import FocusedSBAnomalyDetector

os.environ['LANG'] = 'en_US'
os.environ['PYOPENGL_PLATFORM'] = 'egl' # Uncommnet this line while running remotely

from agent.consistency.observation import ScienceBirdsObservation
from agent.consistency.focused_anomaly_detector import *
import settings
import pickle
import pytest
from os import path

# Constants for ScienceBirds
CP_NON_NOVEL_OBS_DIR = path.join(settings.ROOT_PATH, 'data', 'cartpole', 'consistency', 'non_novel')
CP_NOVEL_OBS_DIR = path.join(settings.ROOT_PATH, 'data', 'cartpole', 'consistency',  'novel')

CP_NON_NOVEL_OBS_FILE_NAME = 'CartPole-v1-non-novel_%d.p'
CP_NOVEL_OBS_FILE_NAME = 'CartPole-v1-novel_%d.p'
CP_NON_NOVEL_OBS = 3
CP_NOVEL_OBS = 3


# Constants for ScienceBirds
SB_NON_NOVEL_OBS_DIR = path.join(settings.ROOT_PATH, 'data', 'science_birds', 'consistency', 'dynamics', 'non_novel')
SB_NOVEL_OBS_DIR = path.join(settings.ROOT_PATH, 'data', 'science_birds', 'consistency', 'dynamics', 'novel')

SB_NON_NOVEL_TESTS = ['level_15_obs.p']
SB_NOVEL_TESTS = ['novelty_2_6_level_15_new_bird_obs.p', 'novelty_2_7_level_15_new_bird_obs.p']

# @pytest.mark.skip("Currently failing.")
def test_UPenn_consistency_cartpole():
    '''
    verify that we can identify novelty for observations of novel problems, and that we don't for non_novel-problems
    '''
    detector = FocusedAnomalyDetector()
    for i in range(CP_NON_NOVEL_OBS):

        obs_output_file = path.join(CP_NON_NOVEL_OBS_DIR, CP_NON_NOVEL_OBS_FILE_NAME % i)  # For debug
        obs = pickle.load(open(obs_output_file, "rb"))
        novelties, novelty_likelihood = detector.detect(obs)
        assert (len(novelties) == 0) # "Non-novel level considered novel (false positive)"
        assert (novelty_likelihood<1)

    for i in range (CP_NOVEL_OBS):
        print("novelty episode %i" % i)
        obs_output_file = path.join(CP_NOVEL_OBS_DIR, CP_NOVEL_OBS_FILE_NAME % i)  # For debug
        obs = pickle.load(open(obs_output_file, "rb"))
        novelties, novelty_likelihood = detector.detect(obs)
        assert(len(novelties)>0) # "Novelty not detected (false negative)"
        assert(novelty_likelihood==1.0)

#@pytest.mark.skip("Skipping science birds for now")
def test_UPenn_consistency_science_birds():
    '''
    verify that we can identify novelty for observations of novel problems, and that we don't for non_novel-problems
    '''
    detector = FocusedSBAnomalyDetector(threshold = 0.3)
    for ob_file in SB_NON_NOVEL_TESTS:
        #load file
        sb_ob : ScienceBirdsObservation = pickle.load(open(path.join(SB_NON_NOVEL_OBS_DIR, ob_file), "rb"))
        novelties, novelty_likelihood = detector.detect(sb_ob)
        assert(len(novelties)==0)
        assert(novelty_likelihood<1)

    for ob_file in SB_NOVEL_TESTS:
        sb_ob : ScienceBirdsObservation = pickle.load(open(path.join(SB_NOVEL_OBS_DIR, ob_file), "rb"))
        novelties, novelty_likelihood = detector.detect(sb_ob)
        assert(len(novelties)>0)
        assert(novelty_likelihood==1)

# Data generation methods - NOT TESTS
@pytest.mark.skip("Generates data for  test_UPenn_consistency_cartpole() - not a real test")
def test_generate_data_for_cartpole():
    import gym
    import agent.gym_hydra_agent

    save_obs = True

    env = gym.make("CartPole-v1")
    cartpole_hydra = agent.gym_hydra_agent.GymHydraAgent(env)

    # Create non_novel obs
    for i in range(CP_NON_NOVEL_OBS):
        cartpole_hydra.observation = cartpole_hydra.env.reset()
        cartpole_hydra.run()  # enough actions to play a level
        if save_obs:
            observation = cartpole_hydra.find_last_obs()
            obs_output_file_name = path.join(CP_NON_NOVEL_OBS_DIR, CP_NON_NOVEL_OBS_FILE_NAME % i)
            obs_output_file = open(obs_output_file_name, "wb")
            pickle.dump(observation, obs_output_file)
            obs_output_file.close()
        print("Created non-novel instance %d" % i)

    # Create novel obs
    # cartpole_hydra.meta_model.constant_numeric_fluents["gravity"] = 19 # Fault injevtion
    env.env.gravity=19
    for i in range (CP_NOVEL_OBS):
        cartpole_hydra.observation = cartpole_hydra.env.reset()
        cartpole_hydra.run()  # enough actions to play a level

        if save_obs:
            observation = cartpole_hydra.find_last_obs()
            obs_output_file_name = path.join(CP_NOVEL_OBS_DIR, CP_NOVEL_OBS_FILE_NAME % i)  # For debug
            obs_output_file = open(obs_output_file_name, "wb")
            pickle.dump(observation, obs_output_file)
            obs_output_file.close()
        print("Created novel instance %d" % i)
    assert(True, "Data generated successfully")
