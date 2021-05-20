
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
from agent.planning.sb_meta_model import *


@pytest.fixture(scope="module")
def launch_science_birds():
    env = sb.ScienceBirds(None,launch=True,config='test_revision.xml')
    yield env
    env.kill()

@pytest.mark.skip("Unsure what the first revision test problems should be.")
def test_science_birds_agent(launch_science_birds):
    env = launch_science_birds
    env.sb_client.set_game_simulation_speed(settings.SB_SIM_SPEED)
    hydra = HydraAgent(env)
    hydra.main_loop() # enough actions to play the first two levels

# this currently runs level 15 followed by its novelty conditions with different birds
if __name__ == '__main__':
    env = sb.ScienceBirds(None, launch=True, config='test_revision.xml')
    test_science_birds_agent(env)


