import pytest

import settings
from agent.planning.polycraft_planning.fixed_planner import FixedPlanPlanner
from agent.polycraft_hydra_agent import *
from agent.planning.polycraft_planning.tasks import *

TEST_PATH = pathlib.Path(settings.ROOT_PATH) / "tests"

@pytest.mark.skip()
def test_generate_data_for_tests():
    ''' Run this to generate the data used by the tests in this test module '''
    # Load levels
    levels = []
    levels_dir_path = pathlib.Path(settings.POLYCRAFT_NON_NOVELTY_LEVEL_DIR)
    for level_file in os.listdir(levels_dir_path):
        if level_file.endswith(".json2") == False:
            levels.append(levels_dir_path / level_file)

    logger.info("starting")
    env = None
    max_iterations = 3
    try:
        env = Polycraft(polycraft_mode=ServerMode.SERVER)
        for i in range(len(levels)):
            test_level = levels[i]
            agent = PolycraftHydraAgent()
            agent.planner = FixedPlanPlanner()

            logger.info(f"Loading level {test_level}...")
            env.init_selected_level(test_level)
            agent.start_level(env)  # Collect trades and recipes
            state = env.get_current_state()
            state, step_cost = agent.do_batch(max_iterations, state, env)
                # Store state file
            with open(TEST_PATH / f"polycraft_state_{i}.p", "wb") as out_file:
                pickle.dump(state, out_file)
    finally:
        logger.info("teardown tests")
        if env is not None:
            env.kill()

@pytest.mark.parametrize('execution_number', range(1))
def test_to_pddl_problem(execution_number):
    state = None
    with open(TEST_PATH / f"polycraft_state_{execution_number}.p" ,"rb") as in_file:
        state = pickle.load(in_file)

    meta_model = PolycraftMetaModel(active_task=PolycraftTask.CRAFT_POGO.create_instance())
    pddl_problem = meta_model.create_pddl_problem(state)

    assert pddl_problem is not None

    PddlProblemExporter().to_file(pddl_problem, TEST_PATH / "poly_prob.pddl")

@pytest.mark.parametrize('execution_number', range(1))
def test_to_pddl_domain(execution_number):
    test_path = pathlib.Path(settings.ROOT_PATH) / "tests"

    state = None
    with open(test_path / f"polycraft_state_{execution_number}.p" ,"rb") as in_file:
        state = pickle.load(in_file)

    meta_model = PolycraftMetaModel(active_task=PolycraftTask.CRAFT_POGO.create_instance())
    pddl_domain = meta_model.create_pddl_domain(state)

    assert pddl_domain is not None

    PddlDomainExporter().to_file(pddl_domain, test_path / "poly_domain.pddl")

@pytest.mark.parametrize('execution_number', range(1))
def test_run_planner(execution_number):
    test_path = pathlib.Path(settings.ROOT_PATH) / "tests"

    state = None
    with open(test_path / f"polycraft_state_{execution_number}.p" ,"rb") as in_file:
        state = pickle.load(in_file)

    # Uncomment to make the planner go faster by making the problem easier
    # state.inventory['4'] = {'item': ItemType.TREE_TAP.value, 'count': 1}
    # state.inventory['5'] = {'item': ItemType.BLOCK_OF_TITANIUM.value, 'count': 2}
    # state.inventory['6'] = {'item': ItemType.DIAMOND_BLOCK.value, 'count': 2}

    planner = PolycraftPlanner()
    plan = planner.make_plan(state)

    assert(len(plan)>0)

@pytest.fixture(scope="module")
def launch_polycraft():
    # Load levels
    levels = get_non_novelty_levels_files()
    logger.info("starting")
    env = Polycraft(polycraft_mode=ServerMode.SERVER)
    agent = PolycraftHydraAgent()

    yield env, agent, levels

    logger.info("teardown tests")
    env.kill()

@pytest.mark.parametrize('execution_number', range(100))
def test_plan_for_non_novelty_levels(launch_polycraft, execution_number):
    ''' Run the fixed planner and observe results '''
    env, agent, levels = launch_polycraft
    test_level = levels[execution_number % len(levels)]  # Choose the level to try now

    logger.info(f"Loading level {test_level}...")
    env.init_selected_level(test_level)
    agent.start_level(env)  # Collect trades and recipes
    state = env.get_current_state()

    planner = PolycraftPlanner()
    plan = planner.make_plan(state)
    # If failed, store state
    if len(plan)==0:
        with open(TEST_PATH / f"failed_init_state_{test_level.name}.p", "wb") as out_file:
            pickle.dump(state, out_file)

    assert(len(plan)>0)

@pytest.mark.parametrize('execution_number', range(100))
def test_plan_for_published_novelty_levels(launch_polycraft, execution_number):
    ''' Run the fixed planner and observe results '''
    env, agent, _ = launch_polycraft
    levels_dir_path = pathlib.Path(settings.POLYCRAFT_NOVELTY_LEVEL_DIR)
    levels = []
    for level_file in os.listdir(levels_dir_path):
        if level_file.endswith(".json2") == False:
            levels.append(levels_dir_path / level_file)
    test_level = levels[execution_number % len(levels)]  # Choose the level to try now


    logger.info(f"Loading level {test_level}...")
    env.init_selected_level(test_level)
    agent.start_level(env)  # Collect trades and recipes
    state = env.get_current_state()

    planner = PolycraftPlanner()
    plan = planner.make_plan(state)
    # If failed, store state
    if len(plan)==0:
        with open(TEST_PATH / f"failed_init_state_{test_level.name}.p", "wb") as out_file:
            pickle.dump(state, out_file)

    assert(len(plan)>0)

# def test_observation():
#     test_path = pathlib.Path(settings.ROOT_PATH) / "tests"
#
#     obs = None
#     with open(test_path / "polycraft_obs_tp_agent.p", "rb") as in_file:
#     # with open(test_path / "polycraft_obs_noop_agent.p", "rb") as in_file:
#         obs = pickle.load(in_file)
#
#     print("Trajectory")
#     fig = plt.figure()
#     ax = fig.add_subplot(projection='3d')
#     visualize_game_map(obs.states[0], ax, set_limits=True)
#     for i in range(len(obs.actions)-1):
#         action = obs.actions[i]
#         success = obs.actions[i].success
#         pre_state = obs.states[i]
#         post_state = obs.states[i+1]
#         if success==True:
#             print("Pre-state: {}".format(_state_entities(pre_state)))
#             print("Action: {}".format(str(action)))
#             print("Post-state: {}".format(_state_entities(post_state)))
#             print(" ")
#         ax.clear()
#         visualize_game_map(post_state, ax, set_limits=True)
#
#
# def _state_entities(state: PolycraftState):
#     str_elements = []
#     steve_pos = str(state.location["pos"])
#     str_elements.append(f'Steve{steve_pos}')
#     for entity_id, entity_attr in state.entities.items():
#         str_elements.append(f'{entity_attr["type"]}{entity_id}{entity_attr["pos"]}')
#
#     return " ".join(str_elements)
#



