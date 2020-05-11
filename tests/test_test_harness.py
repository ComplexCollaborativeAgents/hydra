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

@pytest.fixture(scope="module")
def launch_science_birds():
    print("starting")
    #remove config files
    cmd = 'cp {}/data/science_birds/level-14.xml {}/00001.xml'.format(str(settings.ROOT_PATH), str(settings.SCIENCE_BIRDS_LEVELS_DIR))
    subprocess.run(cmd, shell=True)
    cmd = 'cp {}/data/science_birds/level-15.xml {}/00002.xml'.format(str(settings.ROOT_PATH), str(settings.SCIENCE_BIRDS_LEVELS_DIR))
    subprocess.run(cmd, shell=True)
    cmd = 'cp {}/data/science_birds/level-16.xml {}/00003.xml'.format(str(settings.ROOT_PATH), str(settings.SCIENCE_BIRDS_LEVELS_DIR))
    subprocess.run(cmd, shell=True)
    env = sb.ScienceBirds(None,launch=True,config=None) #run the experiment
    yield env
    print("teardown tests")
    env.kill()

@pytest.mark.skipif(settings.HEADLESS==True, reason="headless does not work in docker")
def test_test_harness(launch_science_birds):
    env = launch_science_birds
    env.sb_client.set_game_simulation_speed(50) #run at max speed as we are just testing all the API calls
    hydra = HydraAgent(env)
    hydra.main_loop()  # enough actions to play the first two levels
