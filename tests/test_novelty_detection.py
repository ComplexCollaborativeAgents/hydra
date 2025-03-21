import worlds.science_birds as sb
import pytest
import sys
from os import path
import settings
import math
import time
import agent.planning.sb_planner as pl
from worlds.science_birds_interface.client.agent_client import GameState

from pprint import pprint
from utils.point2D import Point2D


import subprocess
import agent.perception.perception as perception
from agent.sb_hydra_agent import SBHydraAgent


@pytest.mark.skip("Generate observations for novelty and non-novelty traces for UPenn")
def test_test_harness(launch_science_birds):
    env = launch_science_birds
    hydra = SBHydraAgent(env)
    hydra.main_loop()  # enough actions to play the first two levels
    assert hydra.consistency_checker.novelty_likelihood == 1

# This is just focused on level 1 novelty detecting unknown objects
# This function will generate observations for each level
if __name__ == '__main__':
    env = sb.ScienceBirds(None,launch=True,config='all_level_1_novelties.xml')
    hydra = SBHydraAgent(env)
    hydra.main_loop()
    env.kill()
#    assert hydra.consistency_checker.novelty_likelihood == 1