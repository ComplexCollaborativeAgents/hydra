import pytest
import matplotlib
import numpy
import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from agent.polycraft_hydra_agent import PolycraftObservation
import settings
import pathlib
import pickle
from agent.planning.pddlplus_parser import *
from agent.planning.polycraft_meta_model import *
import time
import matplotlib.pyplot as plt
import numpy as np
from agent.planning.pddlplus_parser import *
from agent.planning.nyx import nyx
from agent.polycraft_hydra_agent import *

TEST_PATH = pathlib.Path(settings.ROOT_PATH) / "tests"

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
        env = Polycraft(launch=True)
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

    meta_model = PolycraftMetaModel()
    pddl_problem = meta_model.create_pddl_problem(state)

    assert pddl_problem is not None

    PddlProblemExporter().to_file(pddl_problem, TEST_PATH / "poly_prob.pddl")

@pytest.mark.parametrize('execution_number', range(1))
def test_to_pddl_domain(execution_number):
    test_path = pathlib.Path(settings.ROOT_PATH) / "tests"

    state = None
    with open(test_path / f"polycraft_state_{execution_number}.p" ,"rb") as in_file:
        state = pickle.load(in_file)

    meta_model = PolycraftMetaModel()
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
# def visualize_game_map(state: PolycraftState, ax=None, set_limits = True):
#     # Fixing random state for reproducibility
#
#     is_3d = True
#     show_cells = True
#     show_steve = True
#     show_entities = True
#     print_results = False
#
#     steve_scatter = None
#     cells_scatter = None
#     entities_scatter = None
#
#     # Create plot if needed
#     if ax is None:
#         fig = plt.figure()
#         if is_3d:
#             ax = fig.add_subplot(projection='3d')
#         else:
#             ax = fig.add_subplot()
#
#     xs = []
#     ys = []
#     zs = []
#     types = []
#     type_to_cells = dict()
#     for cell_id, cell_attr in state.game_map.items():
#         x, y, z = cell_id.split(",")
#         xs.append(int(x))
#         ys.append(int(y))
#         zs.append(int(z))
#         cell_type = cell_attr['name']
#         types.append(cell_type)
#
#         if cell_type not in type_to_cells:
#             type_to_cells[cell_type] = [[],[],[]]
#
#         type_to_cells[cell_type][0].append(int(x))
#         type_to_cells[cell_type][1].append(int(y))
#         type_to_cells[cell_type][2].append(int(z))
#
#     if set_limits:
#         ax.set_xlim(min(xs), max(xs))
#         if is_3d:
#             ax.set_ylim(min(ys), max(ys))
#             ax.set_zlim(min(zs), max(zs))
#         else:
#             ax.set_ylim(min(zs), max(zs))
#
#     if show_cells:
#         if print_results:
#             print(f'x values:{min(xs)}, {max(xs)}')
#             print(f'y values:{min(ys)}, {max(ys)}')
#             print(f'z values:{min(zs)}, {max(zs)}')
#             print('Cell types:')
#             for cell_type in set(types):
#                 print(f'\t {cell_type}')
#
#         type_to_marker = {
#             "minecraft:bedrock": "$b$",
#             "polycraft:plastic_chest": "$p$",
#             "minecraft:log": "$l$",
#             "minecraft:wooden_door": "$d$",
#             "minecraft:air": "$a$",
#             "polycraft:tree_tap": "$t$",
#             "minecraft:crafting_table": "$c$",
#             "polycraft:block_of_platinum": "$b$",
#             "minecraft:diamond_ore": "$D$",
#         }
#
#         for cell_type, cells in type_to_cells.items():
#             if cell_type=="minecraft:air":
#                 continue
#             m = type_to_marker[cell_type]
#             xs = cells[0]
#             ys = cells[1]
#             zs = cells[2]
#
#             if is_3d:
#                 cells_scatter = ax.scatter(xs, ys, zs, marker=m)
#             else:
#                 cells_scatter = ax.scatter(xs, zs, marker=m)
#
#
#     # Steve's location
#     if show_steve:
#         player_obj = state.location
#         steve_x = player_obj['pos'][0]
#         steve_y = player_obj['pos'][1]
#         steve_z = player_obj['pos'][2]
#         if print_results:
#             print(f'Steve at {steve_x, steve_y, steve_z}')
#
#         if is_3d:
#             steve_scatter = ax.scatter([steve_x], [steve_y], [steve_z], marker="*")
#         else:
#             steve_scatter = ax.scatter([steve_x], [steve_z], marker="*")
#
#     # Other entities
#     entities_scatters = []
#     if show_entities:
#         for entity_id, entity_attr in state.entities.items():
#             x, y, z = entity_attr['pos']
#             if print_results:
#                 print(f"Entity {entity_id} at {x,y,z}")
#
#             if is_3d:
#                 scatter= ax.scatter([x], [y], [z], marker="$E$")
#             else:
#                 scatter = ax.scatter([x], [z], marker="$E$")
#             entities_scatters.append(scatter)
#
#     ax.set_xlabel('X Label')
#     if is_3d:
#         ax.set_ylabel('Y Label')
#         ax.set_zlabel('Z Label')
#     else:
#         ax.set_ylabel('Z Label')
#
#     plt.show()
#
#
#     return ax, steve_scatter, cells_scatter, entities_scatters


