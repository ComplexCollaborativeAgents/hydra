
import worlds.science_birds as sb
import pytest
import sys
from os import path
import settings
import math
import time
import agent.planning.planner as pl
from client.agent_client import GameState

from pprint import pprint
from utils.point2D import Point2D


import subprocess
import agent.perception.perception as perception

@pytest.mark.skipif(settings.HEADLESS==True,reason="headless does not work in docker")
def test_science_birds():
    print("starting")

    if sys.platform == 'darwin':
        cmd = 'cp data/science_birds/level-00-original.xml bin/ScienceBirds_MacOS.app/Contents/Resources/Data/StreamingAssets/Levels'
        subprocess.run(cmd, shell=True)
    else:
        cmd = 'cp {}/data/science_birds/level-00-original.xml {}/bin/ScienceBirds_Linux/sciencebirds_linux_Data/StreamingAssets/Levels'.format(str(settings.ROOT_PATH), str(settings.ROOT_PATH))
        subprocess.run(cmd, shell=True)
    env = sb.ScienceBirds(15)

    state = env.get_current_state()
    # env.serialize_current_state(path.join(settings.ROOT_PATH, 'data', 'science_birds', 'serialized_levels', 'level-00.p'))
    # loaded_serialized_state = env.load_from_serialized_state(path.join(settings.ROOT_PATH, 'data', 'science_birds', 'serialized_levels', 'level-00.p'))
    # assert isinstance(state, sb.SBState)
    # assert state.objects == loaded_serialized_state.objects
    print('\nAll Objects: ')

    print(state.objects)
    # assert(len(state.objects) == 7)
    # assert('id' in state.objects[0].keys() and
    #        'type' in state.objects[0].keys() and
    #        'yindex' in state.objects[0].keys() and
    #        'colormap' in state.objects[0].keys())

    p = perception.Perception()
    p.process_state(state)
    # assert(len(state.objects) == 5)
    # assert('type' in state.objects[0].keys() and
    #        'bbox' in state.objects[0].keys())

    # print(str(env.cur_sling.bottom_right))

    planner = pl.Planner()
    planner.write_problem_file(state.translate_state_to_pddl())
    time.sleep(1)

    # env.sb_client.tp.estimate_launch_point(env.cur_sling, Point2D(540,355))

    # state.sling.width,state.sling.height = state.sling.height,state.sling.width
    ref_point = env.tp.get_reference_point(state.sling)
    actions_from_plan = planner.get_plan_actions()
    release_point_from_plan = env.tp.find_release_point(state.sling, math.radians(actions_from_plan[0][1] * 1.00))
    action = sb.SBAction(release_point_from_plan.X, release_point_from_plan.Y, 1000, ref_point.X, ref_point.Y)
    print("\n\n ACTION EXECUTED: " + str(actions_from_plan[0]))
    state, reward, done = env.act(action)

    game_state = GameState.PLAYING

    action_idx = 1
    wtf_counter = 0

    while True:

        # env.sb_client.do_screenshot()
        # env.sb_client.get_ground_truth_with_screenshot()
        state = env.get_current_state()
        game_state = env.sb_client.get_game_state()
        # p.process_state(state)

        if game_state == GameState.WON:
            print("\nWINNNNNNNNN\n==========================\n")
            break
        elif game_state == GameState.LOST:
            print("\nBOOOOOOOOOO\n==========================\n")
            break
        elif game_state == GameState.PLAYING:
            time.sleep(1)

            if action_idx >= len(actions_from_plan):
                print("\nexecuted last action...\n")
                if wtf_counter == 10:
                    break
                wtf_counter += 1
                continue

            p.process_state(state)
            # for a_i in planner.get_plan_actions():
            release_point_from_plan = env.tp.find_release_point(state.sling, math.radians(actions_from_plan[action_idx][1]*1.00))
            action = sb.SBAction(release_point_from_plan.X, release_point_from_plan.Y, 1000, ref_point.X, ref_point.Y)
            print("\n\n ACTION EXECUTED: " + str(actions_from_plan[action_idx]))
            state,reward,done = env.act(action)
            action_idx += 1
            assert isinstance(state,sb.SBState)
            print("STILL PLAYING?!\n==========================\n")

        else:
            print("WHATS HAPPENING?!\n==========================\n")
            if wtf_counter == 3:
                break
            wtf_counter += 1
            time.sleep(5)


    # time.sleep(15)
    env.kill()




def test_state_serialization():
    # state = SBState.serialize_current_state(path.join(settings.ROOT_PATH, 'data', 'science_birds', 'serialized_levels', 'level-00.p'))
    loaded_serialized_state = sb.SBState.load_from_serialized_state(path.join(settings.ROOT_PATH, 'data', 'science_birds', 'serialized_levels', 'level-00.p'))
    assert isinstance(loaded_serialized_state, sb.SBState)
    # assert state.objects == loaded_serialized_state.objects