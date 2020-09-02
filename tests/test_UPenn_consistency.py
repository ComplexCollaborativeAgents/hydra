import worlds.science_birds as sb
import pytest
from agent.consistency.observation import ScienceBirdsObservation
from agent.consistency.focused_anomaly_detector import *
from os import path

NON_NOVEL_OBS_DIR = path.join(settings.ROOT_PATH, 'data','science_birds','consistency','dynamics','non_novel')
NOVEL_OBS_DIR = path.join(settings.ROOT_PATH, 'data','science_birds','consistency','dynamics','novel')

NON_NOVEL_TESTS = ['level_15_obs.p']
NOVEL_TESTS = ['novelty_2_6_level_15_new_bird_obs.p','novelty_2_7_level_15_new_bird_obs.p']

@pytest.mark.skip("Test currently fails")
def test_UPenn_consistency():
    '''
    verify that we can identify novelty for observations of novel problems, and that we don't for non-novel-problems
    '''
    detector = FocusedAnomalyDetector(threshold = 0.3)
    for ob_file in NON_NOVEL_TESTS:
        #load file
        sb_ob : ScienceBirdsObservation = pickle.load(open(path.join(NON_NOVEL_OBS_DIR, ob_file), "rb"))
        novelties = detector.detect(sb_ob)
        assert(len(novelties)==0)

    for ob_file in NOVEL_TESTS:
        sb_ob : ScienceBirdsObservation = pickle.load(open(path.join(NOVEL_OBS_DIR, ob_file), "rb"))
        novelties = detector.detect(sb_ob)
        assert(len(novelties)>0)
