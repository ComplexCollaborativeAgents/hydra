import logging
from worlds.polycraft_world import Polycraft
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
    test_level = path.join(settings.ROOT_PATH, "bin", "pal", "pogo_100_PN", "POGO_L00_T01_S01_X0100_U9999_V0_G00000_I0020_N0.json")
    dumps_path = pathlib.Path(settings.ROOT_PATH) / "data" / "polycraft" / "dumps"
    env.init_selected_level(test_level)

    agent.explore_level(env) # Collect recipes and trades

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

        observation.actions_success.append(env.last_cmd_success)
        observation.rewards.append(-step_cost)

        with open(dumps_path / "test_polycraft_obs_{}.p".format(i),"wb") as out:
            pickle.dump(observation, out)

        logger.info("Post action state: {}".format(str(after_state)))
        logger.info("Step cost: {}".format(step_cost))

        state = after_state

        if state.terminal==True:
            break

if __name__ == '__main__':
    logger.info("starting to generate observations")
    env = Polycraft(launch=True)
    try:
        obs_generator(env, agent=PolycraftManualAgent())
    finally:
        env.kill()
    #
    #
    #
    #
    # dumps_path = pathlib.Path(settings.ROOT_PATH) / "data" / "polycraft" / "dumps"
    # with open(dumps_path / "test_polycraft_obs_49.p", "rb") as in_file:
    #     obs = pickle.load(in_file)
    #
    # print(obs)
    # import agent.repair.action_learner as learner
    # learner.process_trajectory(obs, PolycraftMetaModel())

    # dumps_path = pathlib.Path(settings.ROOT_PATH) / "data" / "polycraft" / "dumps"
    # #
    # world_state = None
    # with open(dumps_path / "test_polycraft_state_1.p", "rb") as in_file:
    #     world_state = pickle.load(in_file)
    #
    # print(world_state)
    # meta_model = PolycraftMetaModel()
    # pddl_problem =meta_model.create_pddl_problem(world_state)
    #
    # PddlProblemExporter().to_file(pddl_problem, dumps_path / "poly-problem.pddl")
    #
    # hydra_agent = PolycraftHydraAgent()
    # hydra_agent.choose_action(world_state)


    #
    # pddl_domain = meta_model.create_pddl_domain(world_state)
    # PddlDomainExporter().to_file(pddl_domain, dumps_path / "poly-domain.pddl")
    #
    # #
    # #
    # # pddl_state = meta_model.create_pddl_state(world_state)
    # # print(pddl_state.to_pddl())