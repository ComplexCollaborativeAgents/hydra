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
    env = sb.ScienceBirds(None,launch=True,config='test_novelty_config.xml') #run a single novelty problem
    yield env
    print("teardown tests")
    env.kill()

@pytest.mark.skipif(settings.HEADLESS==True, reason="headless does not work in docker")
def test_test_harness(launch_science_birds):
    env = launch_science_birds
    hydra = HydraAgent(env)
    hydra.main_loop()  # enough actions to play the first two levels
    assert hydra.consistency_checker.novelty_likelihood == 1
