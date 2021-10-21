import logging
from worlds.polycraft_world import Polycraft, BlockType
from agent.planning.pddlplus_parser import PddlProblemExporter,PddlDomainParser, PddlDomainExporter
from agent.planning.polycraft_meta_model import PolycraftMetaModel
from agent.polycraft_hydra_agent import *
import pickle
import os.path as path
import pathlib
import settings
logging.basicConfig(format='%(name)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Polycraft")

def obs_generator(env:Polycraft, agent: PolycraftHydraAgent = PolycraftHydraAgent()):
    ''' Generate observations for learning polycraft world '''
    dumps_path = pathlib.Path(settings.ROOT_PATH) / "data" / "polycraft" / "dumps"

    state = env.get_current_state()

    # Perform a set of actions
    observation = PolycraftObservation()
    max_iterations = 1000
    for i in range(max_iterations):
        with open(dumps_path / "test_polycraft_state_{}.p".format(i),"wb") as out:
            pickle.dump(state, out)
        action = agent.choose_action(state)
        with open(dumps_path / "test_polycraft_action_{}.p".format(i),"wb") as out:
            pickle.dump(action, out)

        logger.info("State[{}]: {}".format(i, str(state)))
        logger.info("Chosen action: {}".format(action))

        observation.states.append(state)
        observation.actions.append(action)

        after_state, step_cost = env.act(action)

        observation.rewards.append(-step_cost)

        with open(dumps_path / "test_polycraft_obs_{}.p".format(i),"wb") as out:
            pickle.dump(observation, out)

        logger.info("Post action state: {}".format(str(after_state)))
        logger.info("Step cost: {}".format(step_cost))

        state = after_state

        if state.terminal==True:
            break

def generate_obs_of_mining_diamonds(env:Polycraft, agent:PolycraftHydraAgent, iteration=0):
    ''' Generate a set of states we obtain after mining a diamond ore block'''
    dumps_path = pathlib.Path(settings.ROOT_PATH) / "data" / "polycraft" / "dumps"

    state = env.get_current_state()

    # Select iron pickaxe
    action = PolySelectItem(ItemType.IRON_PICKAXE.value)
    after_state, step_cost = agent.do(action, env)
    assert(action.success)
    assert(after_state.get_selected_item()==ItemType.IRON_PICKAXE.value)
    state = after_state

    # Mine
    diamond_cells = state.get_cells_of_type(BlockType.DIAMOND_ORE.value, only_accessible=True)
    cell = diamond_cells[0]
    action = PolyBreakAndCollect(cell)
    with open(dumps_path / "data_polycraft_action_{}.p".format(iteration),"wb") as out:
        pickle.dump(action, out)

    logger.info("State[{}]: {}".format(iteration, str(state)))
    logger.info("Chosen action: {}".format(action))

    agent.do(action, env)

    with open(dumps_path / "data_polycraft_obs_{}.p".format(i),"wb") as out:
        pickle.dump(agent.current_observation, out)

def generate_obs_of_no_op(env, agent, iterations):
    state = env.get_current_state()
    for i in range(iterations):
        action = PolyNoAction()
        next_state, state_cost = agent.do(action, env)
        state = next_state

    dumps_path = pathlib.Path(settings.ROOT_PATH) / "data" / "polycraft" / "dumps"
    with open(dumps_path / "no_op_data_polycraft_obs.p".format(i),"wb") as out:
        pickle.dump(agent.current_observation, out)


if __name__ == '__main__':
    test_level = path.join(settings.ROOT_PATH, "bin", "pal", "pogo_100_PN", "POGO_L00_T01_S01_X0100_U9999_V0_G00000_I0020_N0.json")
    logger.info("starting to generate observations")
    try:
        env = Polycraft(launch=True)
        test_level = path.join(test_level)
        for i in range(3):
            env.init_selected_level(test_level)
            agent = PolycraftManualAgent()
            agent.start_level(env) # Collect recipes and trades
            generate_obs_of_mining_diamonds(env, agent=agent, iteration=i)
    finally:
        env.kill()