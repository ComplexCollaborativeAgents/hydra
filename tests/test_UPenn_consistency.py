import worlds.science_birds as sb
import pytest
from os import path

NON_NOVEL_OBS_DIR = path.join(settings.ROOT_PATH, 'data','science_birds','consistency','dynamics','non_novel')
NOVEL_OBS_DIR = path.join(settings.ROOT_PATH, 'data','science_birds','consistency','dynamics','novel')

NON_NOVEL_TESTS = ['level_15_obs.p']
NOVEL_TESTS = ['novelty_2_6_level_15_new_bird_obs.p','novelty_2_7_level_15_new_bird_obs.p']

def test_UPenn_consistency():
    '''
    verify that we can identify novelty for observations of novel problems, and that we don't for non-novel-problems
    '''
    for ob in NON_NOVEL_TESTS:
        #load file
        res = #call function
        assert(res is None)
    for obs in NOVEL_TESTS:
        res =
        assert(res is not None)
