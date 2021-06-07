from agent.consistency.trace_visualizer import animate_expected, animate_observed
from agent.hydra_agent import *
from agent.planning.planner import *
from matplotlib import pyplot as plt
from agent.perception.perception import *
from agent.repairing_hydra_agent import RepairingHydraSBAgent
from tests.test_utils import create_logger
from os import listdir
from agent.consistency import trace_visualizer as trace_visualizer
logger = create_logger("test_sb_visualizer")
PRECISION = 1

# Constants for ScienceBirds
SB_NON_NOVEL_OBS_DIR = path.join(settings.ROOT_PATH, 'data', 'science_birds', 'consistency', 'non_novel')
SB_NOVEL_OBS_DIR = path.join(settings.ROOT_PATH, 'data', 'science_birds', 'consistency', 'novel')
TEST_DATA_DIR = path.join(settings.ROOT_PATH, 'data', 'science_birds', 'tests')

SB_NON_NOVEL_TESTS = listdir(SB_NON_NOVEL_OBS_DIR)
SB_NOVEL_TESTS = listdir(SB_NOVEL_OBS_DIR)

logger = create_logger("test_sb_visualizer")



def test_animate_single_shot():
    config_file = 'test_repair_wood_health.xml'
    expected_gif_file = "expected_trace.gif"
    observed_gif_file = "observed_trace.gif"


    # Setup
    logger.info("Starting ScienceBirds")
    try:
        env = sb.ScienceBirds(None,launch=True,config=config_file)
        hydra = RepairingHydraSBAgent(env)
        meta_model = hydra.meta_model

        hydra.run_next_action()
        observation = hydra.find_last_obs()
        animate_observed(path.join(TEST_DATA_DIR, observed_gif_file),
                         observation, meta_model=meta_model)
        animate_expected(path.join(TEST_DATA_DIR, expected_gif_file),
                         observation, meta_model=meta_model)
    finally:
        # Teardown
        if env is not None:
            env.kill()
            logger.info("Ending ScienceBirds")


def test_animate_observed_and_expected():
    ''' test the animation functionality of the trace visualizer '''
    meta_model = ScienceBirdsMetaModel()
    samples = 10

    for i in range(samples):
        obs_file = SB_NON_NOVEL_TESTS[i]
        sb_ob = pickle.load(open(path.join(SB_NON_NOVEL_OBS_DIR, obs_file), "rb"))

        logger.info("Creating animation for the trace of obs {}".format(i))
        animate_observed(path.join(TEST_DATA_DIR, "animation-observed-{}.gif".format(i)),
                         sb_ob, meta_model=meta_model)

        logger.info("Creating animation for the expected trace of obs {}".format(i))
        animate_expected(path.join(TEST_DATA_DIR, "animation-expected-{}.gif".format(i)),
                         sb_ob, meta_model=meta_model)


def test_animate_observed_and_expected():
    ''' test the animation functionality of the trace visualizer '''
    meta_model = ScienceBirdsMetaModel()
    samples = 10

    for i in range(samples):
        obs_file = SB_NON_NOVEL_TESTS[i]
        sb_ob = pickle.load(open(path.join(SB_NON_NOVEL_OBS_DIR, obs_file), "rb"))
        logger.info("Creating animation for the trace of obs {}".format(i))
        trace_visualizer.animate_observed(path.join(TEST_DATA_DIR,"animation-observed-{}.gif".format(i)), sb_ob, meta_model=meta_model)

        logger.info("Creating animation for the expected trace of obs {}".format(i))
        trace_visualizer.animate_expected(path.join(TEST_DATA_DIR,"animation-expected-{}.gif".format(i)), sb_ob, meta_model=meta_model)

