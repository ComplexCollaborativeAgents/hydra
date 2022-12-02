from agent.polycraft_hydra_agent import *
import pickle
import pathlib
import settings
logging.basicConfig(format='%(name)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Polycraft")


def generate_obs_of_no_op(env, agent, iterations):
    state = env.get_current_state()
    for i in range(iterations):
        action = PolyNoAction()
        next_state, state_cost = agent.do(action, env)
        state = next_state

    dumps_path = pathlib.Path(settings.ROOT_PATH) / "data" / "polycraft" / "dumps"
    with open(dumps_path / "no_op_data_polycraft_obs.p".format(i),"wb") as out:
        pickle.dump(agent.current_observation, out)


def trajectories_to_pddl_files(dump_path = pathlib.Path(settings.ROOT_PATH) / "data" / "polycraft" / "dumps", num_of_obs = 3):
    ''' Analizes an obseration and outputs PDDL state and action files corresponding to the observed trajectories '''
    dumps_path = pathlib.Path(settings.ROOT_PATH) / "data" / "polycraft" / "dumps"
    meta_model = PolycraftMetaModel(active_task=PolycraftTask.CRAFT_POGO.create_instance())
    for i in range(num_of_obs):
        with open(dumps_path / f"polycraft_obs_actions_{i}.pddl", "w") as actions_file:
            with open(dumps_path / f"polycraft_obs_{i}.p", "rb") as in_file:
                obs = pickle.load(in_file)
                for j in range(len(obs.actions)):
                    state = obs.states[j]
                    PddlProblemExporter().to_file(meta_model.create_pddl_problem(state),
                                                  output_file_name=dumps_path / f"polycraft_state_{i}_{j}.pddl")
                    action = obs.actions[j]
                    actions_file.write(f"{action}\n")
                state = obs.states[-1]
                PddlProblemExporter().to_file(meta_model.create_pddl_problem(state), output_file_name=dumps_path / f"polycraft_state_{i}_{len(obs.states)-1}.pddl")


if __name__ == '__main__':
    dumps_path = pathlib.Path(settings.ROOT_PATH) / "data" / "polycraft" / "dumps"
    iterations_per_level = 3
    num_of_levels = 3
    levels = get_non_novelty_levels_files()
    if len(levels)<num_of_levels:
        num_of_levels = len(levels)

    for i in range(num_of_levels):
        test_level = levels[i]
        logger.info(f"starting to generate observations for level {test_level.name}")
        try:
            env = Polycraft(polycraft_mode=ServerMode.SERVER)
            for j in range(iterations_per_level):
                env.init_selected_level(test_level)
                agent = PolycraftHydraAgent()
                agent.start_level(env) # Collect recipes and trades
                state = env.get_current_state()
                agent.do_batch(10, state, env)
                with open(dumps_path / f"polycraft_obs_{test_level.name}_{j}.p", "wb") as out:
                    pickle.dump(agent.current_observation, out)
        finally:
            env.kill()