
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
from agent.planning.sb_meta_model import *

@pytest.fixture(scope="module")
def launch_science_birds():
    print("starting")
    #remove config files

    ### - test_revision2.xml
    ### comparing the wood novelty for level 2 type 3 and type 4
    # cmd = 'cp {}/data/science_birds/level-20.xml {}/00001.xml'.format(str(settings.ROOT_PATH), str(settings.SCIENCE_BIRDS_LEVELS_DIR))
    # subprocess.run(cmd, shell=True)
    # cmd = 'cp {}/data/science_birds/level-20-novel-wood.xml {}/../../../novelty_level_2/type8/Levels/00001.xml'.format(str(settings.ROOT_PATH), str(settings.SCIENCE_BIRDS_LEVELS_DIR))
    # subprocess.run(cmd, shell=True)
    # cmd = 'cp {}/data/science_birds/level-20-novel-wood.xml {}/../../../novelty_level_2/type9/Levels/00001.xml'.format(str(settings.ROOT_PATH), str(settings.SCIENCE_BIRDS_LEVELS_DIR))
    # subprocess.run(cmd, shell=True)

    ###  - test_revision3.xml
    ### displays all types of novel wood obejcts from level 2 (some novel objects can only be used with certain novelty types)
    # cmd = 'cp {}/data/science_birds/level-19-novel-wood.xml {}/../../../novelty_level_2/type8/Levels/00001.xml'.format(str(settings.ROOT_PATH), str(settings.SCIENCE_BIRDS_LEVELS_DIR))
    # subprocess.run(cmd, shell=True)
    # cmd = 'cp {}/data/science_birds/level-19-novel-wood-2.xml {}/../../../novelty_level_2/type8/Levels/00002.xml'.format(str(settings.ROOT_PATH), str(settings.SCIENCE_BIRDS_LEVELS_DIR))
    # subprocess.run(cmd, shell=True)
    # cmd = 'cp {}/data/science_birds/level-19-novel-wood-3.xml {}/../../../novelty_level_2/type8/Levels/00003.xml'.format(str(settings.ROOT_PATH), str(settings.SCIENCE_BIRDS_LEVELS_DIR))
    # subprocess.run(cmd, shell=True)
    # cmd = 'cp {}/data/science_birds/level-19-novel-wood-4.xml {}/../../../novelty_level_2/type8/Levels/00004.xml'.format(str(settings.ROOT_PATH), str(settings.SCIENCE_BIRDS_LEVELS_DIR))
    # subprocess.run(cmd, shell=True)
    # cmd = 'cp {}/data/science_birds/level-19-novel-wood-5.xml {}/../../../novelty_level_2/type8/Levels/00005.xml'.format(str(settings.ROOT_PATH), str(settings.SCIENCE_BIRDS_LEVELS_DIR))
    # subprocess.run(cmd, shell=True)
    # cmd = 'cp {}/data/science_birds/level-19-novel-wood-6.xml {}/../../../novelty_level_2/type8/Levels/00006.xml'.format(str(settings.ROOT_PATH), str(settings.SCIENCE_BIRDS_LEVELS_DIR))
    # subprocess.run(cmd, shell=True)

    ###  - test_revision4.xml
    ### displays all different shapes of wood blocks
    ### Don't copy
    cmd = 'cp {}/data/science_birds/level-19.xml {}/novelty_level_0/type2/Levels/00001.xml'.format(str(settings.ROOT_PATH), str(settings.SCIENCE_BIRDS_LEVELS_DIR))
    subprocess.run(cmd, shell=True)
    cmd = 'cp {}/data/science_birds/level-19-2.xml {}/novelty_level_0/type2/Levels/00002.xml'.format(str(settings.ROOT_PATH), str(settings.SCIENCE_BIRDS_LEVELS_DIR))
    subprocess.run(cmd, shell=True)
    cmd = 'cp {}/data/science_birds/level-19-3.xml {}/novelty_level_0/type2/Levels/00003.xml'.format(str(settings.ROOT_PATH), str(settings.SCIENCE_BIRDS_LEVELS_DIR))
    subprocess.run(cmd, shell=True)
    cmd = 'cp {}/data/science_birds/level-19-4.xml {}/novelty_level_0/type2/Levels/00004.xml'.format(str(settings.ROOT_PATH), str(settings.SCIENCE_BIRDS_LEVELS_DIR))
    subprocess.run(cmd, shell=True)

    env = sb.ScienceBirds(None, launch=True, config='test_revision4.xml')
    yield env
    print("teardown tests")
    env.kill()

@pytest.mark.skip("Unsure what the first revision test problems should be.")
def test_science_birds_agent(launch_science_birds):
    env = launch_science_birds
    env.sb_client.set_game_simulation_speed(settings.SB_SIM_SPEED)
    hydra = SBHydraAgent(env)
    hydra.main_loop() # enough actions to play the first two levels
    assert len(set([o for o in hydra.observations if o.reward > 0])) == 3 # ensure we have 6 shots that hit things
    assert sum(1 for x in hydra.completed_levels if x) == 3 # We pass one level

