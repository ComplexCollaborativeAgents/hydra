
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
  #  cmd = 'rm {}/../../../../*.xml'.format(str(settings.SCIENCE_BIRDS_LEVELS_DIR),                                                                         str(settings.SCIENCE_BIRDS_LEVELS_DIR))
  #  subprocess.run(cmd, shell=True)
    cmd = 'cp {}/data/science_birds/level-14.xml {}/00001.xml'.format(str(settings.ROOT_PATH), str(settings.SCIENCE_BIRDS_LEVELS_DIR))
    subprocess.run(cmd, shell=True)
    cmd = 'cp {}/data/science_birds/level-15.xml {}/00002.xml'.format(str(settings.ROOT_PATH), str(settings.SCIENCE_BIRDS_LEVELS_DIR))
    subprocess.run(cmd, shell=True)
    cmd = 'cp {}/data/science_birds/level-16.xml {}/00003.xml'.format(str(settings.ROOT_PATH), str(settings.SCIENCE_BIRDS_LEVELS_DIR))
    subprocess.run(cmd, shell=True)
    env = sb.ScienceBirds(None,launch=False)
    yield env
    print("teardown tests")
    env.kill()

@pytest.mark.skipif(settings.HEADLESS==True, reason="headless does not work in docker")
def test_science_birds_agent(launch_science_birds):
    env = launch_science_birds
    hydra = HydraAgent(env)
    hydra.main_loop(10) # enough actions to play the first two levels
    scores = env.get_all_scores()
    assert len([x for x in scores if x > 0]) == 3 # solved two problems


@pytest.mark.skipif(True, reason="headless does not work in docker")
def test_science_birds(launch_science_birds):
    env = launch_science_birds
    env.init_selected_level(1)
    state = env.get_current_state()
    print(state.objects)
    assert (len(state.objects) == 7)
    assert ('id' in state.objects[0].keys() and
            'type' in state.objects[0].keys() and
            'yindex' in state.objects[0].keys() and
            'colormap' in state.objects[0].keys())

    p = perception.Perception()
    p.process_state(state)
    assert (len(state.objects) == 5)
    assert ('type' in state.objects[0].keys() and
            'bbox' in state.objects[0].keys())
    count = 0
    state.serialize_current_state( # copy to the other data directory later
        path.join(settings.ROOT_PATH, 'tmp',  'dx_test_{}.p'.format(count)))

    # print(str(env.cur_sling.bottom_right))

    planner = pl.Planner()
    planner.write_problem_file(state.translate_state_to_pddl())

    ref_point = env.tp.get_reference_point(state.sling)
    #release_point_from_plan = env.tp.find_release_point(state.sling, 0.174533) # 10 degree launch
    release_point_from_plan = env.tp.find_release_point(state.sling, math.radians(planner.get_plan_actions()[0][1]))
    action = sb.SBShoot(release_point_from_plan.X, release_point_from_plan.Y, 3000, ref_point.X, ref_point.Y)

    state, reward = env.act(action)
    assert len(env.intermediate_states) > 1
    # some objects should be destroyed by the last state
    assert len(env.intermediate_states[0].objects) > len(env.intermediate_states[-1].objects)
    for s in env.intermediate_states:
        count+=1
        s.serialize_current_state(  # copy to the other data directory later
            path.join(settings.ROOT_PATH, 'tmp', 'dx_test_{}.p'.format(count)))
    assert isinstance(state, sb.SBState)

    count += 1
    state.serialize_current_state( # copy to the other data directory later
        path.join(settings.ROOT_PATH, 'tmp',  'dx_test_{}.p'.format(count)))

    assert reward > 0


@pytest.mark.skipif(True, reason="This functionality is captured in the science birds agent test")
def test_multi_shot(launch_science_birds):
    print('\nAll Objects: ')
    env = launch_science_birds
    env.init_selected_level(2)
    state = env.get_current_state()
    print(state.objects)
    assert(len(state.objects) == 15) # we have loaded the right problem
    p = perception.Perception()
    p.process_state(state)

    planner = pl.Planner()
    planner.write_problem_file(state.translate_state_to_pddl())


    # env.sb_client.tp.estimate_launch_point(env.cur_sling, Point2D(540,355))

    # state.sling.width,state.sling.height = state.sling.height,state.sling.width
    ref_point = env.tp.get_reference_point(state.sling)

    actions_from_plan = []

    counter_syntax = 0

    while True:
        actions_from_plan = planner.get_plan_actions()
        if len(actions_from_plan) > 0:
            if actions_from_plan[0][1] != -999:
                print("\nFound PDDL+ actions, executing...\n")
                break
            elif actions_from_plan[0][0] == "out of memory":
                print("\nPDDL+ planner ran out of memory...\n")
                env.kill()
                return
        else:
            print("\nno actions to execute.. \nprobable cause: PDDL+ Syntax Error \nreplanning...\n")
            # continue
        counter_syntax += 1
        if counter_syntax >= 5:
            env.kill()
            return

    assert(len(actions_from_plan) == 3)
    release_point_from_plan = env.tp.find_release_point(state.sling, math.radians(actions_from_plan[0][1] * 1.00))
    action = sb.SBAction(release_point_from_plan.X, release_point_from_plan.Y, 1000, ref_point.X, ref_point.Y)
    print("action executed: " + str(actions_from_plan[0]))
    state, reward = env.act(action)

    game_state = GameState.PLAYING

    action_idx = 1
    wtf_counter = 0

    while True:
        state = env.get_current_state()
        game_state = env.sb_client.get_game_state()
        # p.process_state(state)

        if game_state.value == GameState.WON.value:
            print("\ngame won!\n")
            break
        elif game_state.value == GameState.LOST.value:
            print("\ngame lost!\n")
            assert(False)
        elif game_state.value == GameState.PLAYING.value:


            if action_idx >= len(actions_from_plan):
                print("\nexecuted last action...")
                if wtf_counter == 10:
                    assert(False)
                wtf_counter += 1
                continue

            p.process_state(state)
            # for a_i in planner.get_plan_actions():
            release_point_from_plan = env.tp.find_release_point(state.sling, math.radians(actions_from_plan[action_idx][1]*1.00))
            action = sb.SBAction(release_point_from_plan.X, release_point_from_plan.Y, 1000, ref_point.X, ref_point.Y)
            print("\nexecuted action: " + str(actions_from_plan[action_idx]))
            state,reward,done = env.act(action)
            action_idx += 1
            assert isinstance(state,sb.SBState)
            # print("STILL PLAYING?!\n==========================\n")

        else:
            print("Unknown state...\n")
            if wtf_counter == 3:
                assert(False)
                break
            wtf_counter += 1

def test_state_serialization():
    # state = SBState.serialize_current_state(path.join(settings.ROOT_PATH, 'data', 'science_birds', 'serialized_levels', 'level-00.p'))
    loaded_serialized_state = sb.SBState.load_from_serialized_state(path.join(settings.ROOT_PATH, 'data', 'science_birds', 'serialized_levels', 'level-00.p'))
    assert isinstance(loaded_serialized_state, sb.SBState)
    # assert state.objects == loaded_serialized_state.objects
