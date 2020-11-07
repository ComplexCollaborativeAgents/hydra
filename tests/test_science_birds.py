
import worlds.science_birds as sb
import pytest
import sys
from os import path
import settings
import math
import time
import agent.planning.planner as pl
from worlds.science_birds_interface.client.agent_client import GameState

from pprint import pprint
from utils.point2D import Point2D


import subprocess
import agent.perception.perception as perception
from agent.hydra_agent import HydraAgent
from agent.planning.pddl_meta_model import *
from agent.repairing_hydra_agent import RepairingHydraSBAgent

logging.basicConfig(format='%(name)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TestSB")

@pytest.fixture(scope="module")
def launch_science_birds():
    logger.info("starting")
    # There is a new method where you can specify a level directory,
    # cmd = 'cp {}/data/science_birds/level-14.xml {}/00001.xml'.format(str(settings.ROOT_PATH), str(settings.SCIENCE_BIRDS_LEVELS_DIR))
    # subprocess.run(cmd, shell=True)
    # cmd = 'cp {}/data/science_birds/level-15.xml {}/00002.xml'.format(str(settings.ROOT_PATH), str(settings.SCIENCE_BIRDS_LEVELS_DIR))
    # subprocess.run(cmd, shell=True)
    # cmd = 'cp {}/data/science_birds/level-16.xml {}/00003.xml'.format(str(settings.ROOT_PATH), str(settings.SCIENCE_BIRDS_LEVELS_DIR))
    # subprocess.run(cmd, shell=True)
    env = sb.ScienceBirds(None,launch=True,config='test_config.xml')
    yield env
    env.kill()
    logger.info("teardown tests")

#@pytest.mark.skipif(False, reason="headless does not work in docker")
def test_science_birds_agent(launch_science_birds):
    env = launch_science_birds
    # env.sb_client.set_game_simulation_speed(settings.SB_SIM_SPEED)
    hydra = HydraAgent(env)
    hydra.main_loop() # enough actions to play the first two levels
    assert len(set([o for o in hydra.observations if o.reward > 0])) == 4 # ensure we have 4 shots that hit things


@pytest.mark.skip("'ScienceBirdsObservation' object has no attribute 'hasUnknownObj'")
def test_science_birds_agent(launch_science_birds):
    env = launch_science_birds
    # env.sb_client.set_game_simulation_speed(settings.SB_SIM_SPEED)
    hydra = RepairingHydraSBAgent(env)
    hydra.main_loop() # enough actions to play the first two levels
    assert len(set([o for o in hydra.observations if o.reward > 0])) == 4 # ensure we have 4 shots that hit things


def test_state_serialization():
    # state = SBState.serialize_current_state(path.join(settings.ROOT_PATH, 'data', 'science_birds', 'serialized_levels', 'level-00.p'))
    loaded_serialized_state = sb.SBState.load_from_serialized_state(path.join(settings.ROOT_PATH, 'data', 'science_birds', 'serialized_levels', 'level-00.p'))
    assert isinstance(loaded_serialized_state, sb.SBState)
    # assert state.objects == loaded_serialized_state.objects
