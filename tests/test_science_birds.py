
import worlds.science_birds as sb
import pytest
import sys
from os import path
import settings
import math
import agent.planning.planner as pl

from pprint import pprint
from utils.point2D import Point2D


import subprocess
import agent.perception.perception as perception

@pytest.fixture(scope="module")
def launch_science_birds():
    print("starting")
    if sys.platform == 'darwin':
        cmd = 'cp data/science_birds/level-00-original.xml bin/ScienceBirds_MacOS.app/Contents/Resources/Data/StreamingAssets/Levels'
        subprocess.run(cmd, shell=True)
    else:
        cmd = 'cp data/science_birds/level-00-original.xml bin/ScienceBirds_Linux/sciencebirds_linux_Data/StreamingAssets/Levels'
        subprocess.run(cmd, shell=True)
    env = sb.ScienceBirds(0)
    yield env
    print("teardown tests")
    env.kill()


@pytest.mark.skipif(settings.HEADLESS==True,reason="headless does not work in docker")
def test_science_birds(launch_science_birds):
    # env.serialize_current_state(path.join(settings.ROOT_PATH, 'data', 'science_birds', 'serialized_levels', 'level-00.p'))
    # loaded_serialized_state = env.load_from_serialized_state(path.join(settings.ROOT_PATH, 'data', 'science_birds', 'serialized_levels', 'level-00.p'))
    # assert isinstance(state, sb.SBState)
    # assert state.objects == loaded_serialized_state.objects
    print('\nAll Objects: ')
    env = launch_science_birds
    state = env.get_current_state()
    print(state.objects)
    assert(len(state.objects) == 7)
    assert('id' in state.objects[0].keys() and
           'type' in state.objects[0].keys() and
           'yindex' in state.objects[0].keys() and
           'colormap' in state.objects[0].keys())

    p = perception.Perception()
    p.process_state(state)
    assert(len(state.objects) == 5)
    assert('type' in state.objects[0].keys() and
           'bbox' in state.objects[0].keys())

    # print(str(env.cur_sling.bottom_right))

    planner = pl.Planner()
    planner.write_problem_file(state.translate_state_to_pddl())

    # env.sb_client.tp.estimate_launch_point(env.cur_sling, Point2D(540,355))

    state.sling.width,state.sling.height = state.sling.height,state.sling.width
    ref_point = env.tp.get_reference_point(state.sling)
    release_point_from_plan = env.tp.find_release_point(state.sling, 0.174533) # 10 degree launch
    #release_point_from_plan = env.tp.find_release_point(state.cur_sling, math.radians(planner.get_plan_actions()[0][1]*1.05))
    # action = sb.SBAction(release_point_from_plan.X, release_point_from_plan.Y,3000) # no idea what the scale of these should be
    action = sb.SBAction(release_point_from_plan.X, release_point_from_plan.Y, 3000, ref_point.X, ref_point.Y)

    state,reward,done = env.act(action)
    assert len(env.intermediate_states) > 1
    # some objects should be destroyed by the last state
    assert len(env.intermediate_states[0].objects) > len(env.intermediate_states[-1].objects)
    assert isinstance(state,sb.SBState)
    assert reward > 0
    assert done == True



def test_state_serialization():
    # state = SBState.serialize_current_state(path.join(settings.ROOT_PATH, 'data', 'science_birds', 'serialized_levels', 'level-00.p'))
    loaded_serialized_state = sb.SBState.load_from_serialized_state(path.join(settings.ROOT_PATH, 'data', 'science_birds', 'serialized_levels', 'level-00.p'))
    assert isinstance(loaded_serialized_state, sb.SBState)
    # assert state.objects == loaded_serialized_state.objects
