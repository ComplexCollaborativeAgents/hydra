import os

from state_prediction.anomaly_detector_fc_multichannel import FocusedSBAnomalyDetector

os.environ['LANG'] = 'en_US'
os.environ['PYOPENGL_PLATFORM'] = 'egl' # Uncommnet this line while running remotely

from agent.consistency.episode_log import ScienceBirdsObservation
from agent.consistency.focused_anomaly_detector import *
import settings
import pickle
import pytest
from os import path, listdir

# Constants for ScienceBirds
CP_NON_NOVEL_OBS_DIR = path.join(settings.ROOT_PATH, 'data', 'cartpole', 'may2021', 'non_novel')
CP_NOVEL_OBS_DIR = path.join(settings.ROOT_PATH, 'data', 'cartpole', 'may2021',  'novel')

CP_NON_NOVEL_OBS_FILE_NAME = 'CartPole-v1-non-novel_%d.p'
CP_NOVEL_OBS_FILE_NAME = 'CartPole-v1-{}{}-novel_{}.p'
CP_NON_NOVEL_OBS = 400
CP_NOVEL_OBS = 50


# Constants for ScienceBirds
SB_NON_NOVEL_OBS_DIR = path.join(settings.ROOT_PATH, 'data', 'science_birds', 'consistency', 'non_novel')
SB_NOVEL_OBS_DIR = path.join(settings.ROOT_PATH, 'data', 'science_birds', 'consistency', 'novel')

SB_NON_NOVEL_TESTS = listdir(SB_NON_NOVEL_OBS_DIR)
SB_NOVEL_TESTS = listdir(SB_NOVEL_OBS_DIR)

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
    detector = FocusedSBAnomalyDetector(threshold = 0.5) # Raising this threshold will reduce false positives

    true_negatives = 0
    true_positives = 0
    false_negatives = 0
    false_positives = 0

    print("Non novel cases:")
    for ob_file in SB_NON_NOVEL_TESTS:
        #load file
        sb_ob : ScienceBirdsObservation = pickle.load(open(path.join(SB_NON_NOVEL_OBS_DIR, ob_file), "rb"))
        novelties, novelty_likelihood = detector.detect(sb_ob)
        print("file={}, novelties found={}, novelty likelihood={}".format(ob_file, len(novelties), novelty_likelihood))
        if not novelties:
            true_negatives += 1
        else:
            false_positives += 1

    print("Novel cases:")
    for ob_file in SB_NOVEL_TESTS:
        sb_ob : ScienceBirdsObservation = pickle.load(open(path.join(SB_NOVEL_OBS_DIR, ob_file), "rb"))
        novelties, novelty_likelihood = detector.detect(sb_ob)
        print("file={}, novelties found={}, novelty likelihood={}".format(ob_file, len(novelties), novelty_likelihood))
        if novelties:
            true_positives += 1
        else:
            false_negatives += 1

    print("tp={}, tn={}, fp={}, fn={}".format(true_positives, true_negatives, false_positives, false_negatives))

    assert(true_positives >= 4)
    assert(true_negatives >= 7)
    assert(false_positives <= 2)
    assert(false_negatives <= 5)

# Data generation methods - NOT TESTS
#@pytest.mark.skip("Generates data for  test_UPenn_consistency_cartpole() - not a real test")
def test_generate_data_for_cartpole():
    import gym
    import agent.gym_hydra_agent

    save_obs = True

    env = gym.make("CartPole-v1")
    cartpole_hydra = agent.gym_hydra_agent.GymHydraAgent(env)

    # Create non_novel obs
    # for i in range(CP_NON_NOVEL_OBS):
    #     cartpole_hydra.observation = cartpole_hydra.env.reset()
    #     cartpole_hydra.run()  # enough actions to play a level
    #     if save_obs:
    #         observation = cartpole_hydra.find_last_obs()
    #         obs_output_file_name = path.join(CP_NON_NOVEL_OBS_DIR, CP_NON_NOVEL_OBS_FILE_NAME % i)
    #         obs_output_file = open(obs_output_file_name, "wb")
    #         pickle.dump(observation, obs_output_file)
    #         obs_output_file.close()
    #     print("Created non-novel instance %d" % i)
    # Create novel obs
    # cartpole_hydra.meta_model.constant_numeric_fluents["gravity"] = 19 # Fault injevtion
    env.env.length=1.4
    assert getattr(env.env, 'length') == 1.4
    for i in range (CP_NOVEL_OBS):
        cartpole_hydra.observation = cartpole_hydra.env.reset()
        cartpole_hydra.run()  # enough actions to play a level

        if save_obs:
            observation = cartpole_hydra.find_last_obs()
            obs_output_file_name = path.join(CP_NOVEL_OBS_DIR, CP_NOVEL_OBS_FILE_NAME.format('length',14,i))  # For debug
            obs_output_file = open(obs_output_file_name, "wb")
            pickle.dump(observation, obs_output_file)
            obs_output_file.close()
        print("Created novel instance %d" % i)
    assert(True, "Data generated successfully")
