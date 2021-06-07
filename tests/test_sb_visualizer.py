import matplotlib.pyplot as plt
import pytest
import time

from agent.consistency.fast_pddl_simulator import *
from agent.repair.sb_repair import BirdLocationConsistencyEstimator, ScienceBirdsConsistencyEstimator
from agent.hydra_agent import *
from agent.planning.model_manipulator import ManipulateInitNumericFluent
from agent.planning.planner import *
from matplotlib import pyplot as plt
from agent.perception.perception import *
import tests.test_utils as test_utils
from tests.test_utils import create_logger
from state_prediction.anomaly_detector_fc_multichannel import FocusedSBAnomalyDetector
from os import listdir
from agent.consistency import trace_visualizer as trace_visualizer
logger = create_logger("test_sb_visualizer")
import time
PRECISION = 1
import numpy as np

# Constants for ScienceBirds
SB_NON_NOVEL_OBS_DIR = path.join(settings.ROOT_PATH, 'data', 'science_birds', 'consistency', 'non_novel')
SB_NOVEL_OBS_DIR = path.join(settings.ROOT_PATH, 'data', 'science_birds', 'consistency', 'novel')
TEST_DATA_DIR = path.join(settings.ROOT_PATH, 'data', 'science_birds', 'tests')

SB_NON_NOVEL_TESTS = listdir(SB_NON_NOVEL_OBS_DIR)
SB_NOVEL_TESTS = listdir(SB_NOVEL_OBS_DIR)

logger = create_logger("test_sb_visualizer")


def test_animate_observed_and_expected():
    ''' test the animation functionality of the trace visualizer '''
    meta_model = ScienceBirdsMetaModel()
    samples = 10

    for i in range(samples):
        obs_file = SB_NON_NOVEL_TESTS[i]
        sb_ob = pickle.load(open(path.join(SB_NON_NOVEL_OBS_DIR, obs_file), "rb"))

        logger.info("Creating animation for the trace of obs {}".format(i))
        obs_state_sequence = sb_ob.get_pddl_states_in_trace(meta_model)
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        trace_animation = trace_visualizer.animate_trace(fig, ax, obs_state_sequence)
        trace_animation.save(path.join(TEST_DATA_DIR,"animation-observed-{}.gif".format(i)))

        logger.info("Creating animation for the expected trace of obs {}".format(i))
        simulator = CachingPddlPlusSimulator()
        sim_trace = simulator.simulate_observed_action(sb_ob.state, sb_ob.action, meta_model)
        sim_state_sequence = [timed_state[0] for timed_state in sim_trace]
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        trace_animation = trace_visualizer.animate_trace(fig, ax, sim_state_sequence)
        trace_animation.save(path.join(TEST_DATA_DIR,"animation-expected-{}.gif".format(i)))

