import logging
import worlds.polycraft_world as poly
from agent.planning.pddlplus_parser import PddlProblemExporter
from agent.planning.polycraft_meta_model import PolycraftMetaModel
from agent.polycraft_hydra_agent import PolycraftHydraAgent
import pickle
import os.path as path
import pathlib
import settings
logging.basicConfig(format='%(name)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Polycraft")


def launch_polycraft(launch_server):
    logger.info("starting")
    env = poly.Polycraft(launch=launch_server)
    yield env
    logger.info("teardown tests")
    env.kill()

def obs_generator(launch_server=True):
    ''' Generate observations for learning polycraft world '''
    env = next(launch_polycraft(launch_server=launch_server))
    hydra = PolycraftHydraAgent()
    test_level = path.join(settings.ROOT_PATH, "bin", "pal", "pogo_100_PN", "POGO_L00_T01_S01_X0100_U9999_V0_G00000_I0020_N0.json")
    dumps_path = pathlib.Path(settings.ROOT_PATH) / "data" / "polycraft" / "dumps"
    env.init_selected_level(test_level)
    state = env.get_current_state()
    logger.info("Initial state: {}".format(str(state)))

    hydra.explore_level(env)

    # Perform a set of actions
    for i in range(50):
        with open(dumps_path / "test_polycraft_state_{}.p".format(i),"wb") as out:
            pickle.dump(state, out)
        action = hydra.choose_action(state)
        with open(dumps_path / "test_polycraft_action_{}.p".format(i),"wb") as out:
            pickle.dump(action, out)
        logger.info("Chose action: {}".format(action))

        after_state, step_cost = env.act(action)
        logger.info("Post action state: {}".format(str(after_state)))
        logger.info("Post action step cost: {}".format(step_cost))

        state = after_state

if __name__ == '__main__':
    # obs_generator(launch_server=True)
    dumps_path = pathlib.Path(settings.ROOT_PATH) / "data" / "polycraft" / "dumps"
    #
    world_state = None
    with open(dumps_path / "test_polycraft_state_1.p", "rb") as in_file:
        world_state = pickle.load(in_file)

    print(world_state)
    meta_model = PolycraftMetaModel()
    pddl_problem =meta_model.create_pddl_problem(world_state)
    PddlProblemExporter().to_file(pddl_problem, dumps_path / "poly.pddl")
    #
    #
    # pddl_state = meta_model.create_pddl_state(world_state)
    # print(pddl_state.to_pddl())