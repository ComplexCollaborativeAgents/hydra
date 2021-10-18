import pickle
import pytest
import settings
import logging
import pathlib
from agent.planning.polycraft_planning.actions import PolyBreakAndCollect
import worlds.polycraft_world as poly
from agent.polycraft_hydra_agent import PolycraftObservation

logging.basicConfig(format='%(name)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TestPolycraftWorld")


@pytest.fixture(scope="module")
def launch_polycraft():
    logger.info("starting")

    env = poly.Polycraft(launch=True)
    yield env
    logger.info("teardown tests")
    env.kill()

def test_break_and_collect(launch_polycraft: poly.Polycraft):
    ''' Connect to polycraft and perform the break and collect action and make sure the inventory includes a new item '''

    env = launch_polycraft
    test_level = pathlib.Path(settings.POLYCRAFT_NON_NOVELTY_LEVEL_DIR) / "POGO_L00_T01_S01_X0100_U9999_V0_G00000_I0020_N0.json"

    runs = 5
    for i in range(runs):
        env.init_selected_level(test_level)
        state = env.get_current_state()
        action = PolyBreakAndCollect("43,17,42")

        after_state, step_cost = env.act(action)

        diff = after_state.diff(state)
        logger.info(f'Diff after action:\n {diff}')
        if "inventory" not in diff:
            print(diff)
        assert("inventory" in diff)

        action = PolyBreakAndCollect("50,17,40")

        state = after_state
        after_state, step_cost = env.act(action)

        diff = after_state.diff(state)
        if "inventory" not in diff:
            print(diff)
        logger.info(f'Diff after action:\n {diff}')
        assert("inventory" in diff)


def test_observation():
    test_path = pathlib.Path(settings.ROOT_PATH) / "tests"

    obs = None
    with open(test_path / "polycraft_obs_tp_agent.p", "rb") as in_file:
    # with open(test_path / "polycraft_obs_noop_agent.p", "rb") as in_file:
        obs = pickle.load(in_file)

    print("Trajectory")
    for i in range(len(obs.actions)-1):
        action = obs.actions[i]
        success = obs.actions_success[i]
        pre_state = obs.states[i]
        post_state = obs.states[i+1]
        if success==True:
            print("Action: {}".format(str(action)))
            diff = post_state.diff(pre_state)
            print("Diff:")
            for diff_item, diff_details in diff.items():
                if type(diff_details)==dict:
                    for diff_sub_item, diff_sub_details in diff_details.items():
                        print(f'\t{diff_item}[{diff_sub_item}]: {diff_sub_details}')
                else:
                    print(f'\t{diff_item}: {diff_details}')

