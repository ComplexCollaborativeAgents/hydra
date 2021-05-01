from matplotlib import pyplot as plt, patches as patches
from agent.planning.pddl_plus import PddlPlusState
import matplotlib

BIRD_MARKER = ".r"
PIG_MARKER = "o"

'''
Plotting a sequence of states, showing where the pigs, platforms, and birds are initially,
and showing the trajectory of the active bird. 
'''
def plot_state_sequence(pddl_state_seq : list, pddl_state: PddlPlusState, ax= None, marker="x"):
    if ax is None:
        _, ax = plt.subplots()
    # plot_pddl_state(pddl_state, ax)

    previous_state = pddl_state
    for intermediate_pddl_state in pddl_state_seq:
        plot_intermediate_state(intermediate_pddl_state, previous_state, ax)
        previous_state = intermediate_pddl_state

    # Set plot area to be a square, so that proportions are right.
    (left, right_x) = plt.xlim()
    (left, right_y) = plt.ylim()
    max_axis = max(right_x, right_y)
    plt.xlim((0, max_axis))
    plt.ylim((0, max_axis))

    plt.show()
    return ax

''' Plot a platform'''
def plot_platform(platform, pddl_state, ax):
    x = pddl_state[("x_platform", platform)]
    y = pddl_state[("y_platform", platform)]
    width = pddl_state[("platform_width", platform)]
    height = pddl_state[("platform_height", platform)]
    x = x - width / 2
    y = y - height / 2
    rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

''' Plot a block '''
def plot_block(block, pddl_state, ax):
    x = pddl_state[("x_block", block)]
    y = pddl_state[("y_block", block)]
    width = pddl_state[("block_width", block)]
    height = pddl_state[("block_height", block)]
    x = x - width / 2
    y = y - height / 2
    rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='b', facecolor='none')
    ax.add_patch(rect)

''' Plot the pigs and birds in the pddl state '''
def plot_pigs_and_birds(modified_birds, modified_pigs, pddl_state : PddlPlusState, ax):
    # plot birds
    x_birds = [pddl_state[("x_bird", bird)] for bird in modified_birds]
    y_birds = [pddl_state[("y_bird", bird)] for bird in modified_birds]
    ax.plot(x_birds, y_birds, BIRD_MARKER, linestyle="")

    x_pigs = [pddl_state[("x_pig", pig)] for pig in modified_pigs]
    y_pigs = [pddl_state[("y_pig", pig)] for pig in modified_pigs]
    ax.plot(x_pigs, y_pigs, PIG_MARKER, linestyle="")

'''
Plotting an intermediate PDDL+ state. Do not plot an object if it did not change '''
def plot_intermediate_state(pddl_state : PddlPlusState, previous_pddl_state: PddlPlusState, ax ):

    modified_fluents = set()
    for numeric_fluent in pddl_state.numeric_fluents:
        if numeric_fluent in previous_pddl_state.numeric_fluents and \
                pddl_state[numeric_fluent]==previous_pddl_state[numeric_fluent]:
            continue
        modified_fluents.add(numeric_fluent)

    # Identified modified objects
    pigs = pddl_state.get_pigs()
    birds = pddl_state.get_birds()
    platforms = pddl_state.get_platforms()
    blocks = pddl_state.get_blocks()

    modified_pigs = set()
    modified_birds = set()
    modified_blocks = set()
    modified_platforms = set()
    for modified_fluent in modified_fluents:
        if len(modified_fluent) == 2:
            if modified_fluent[1] in pigs:
                modified_pigs.add(modified_fluent[1])
            elif modified_fluent[1] in birds:
                modified_birds.add(modified_fluent[1])
            elif modified_fluent[1] in blocks:
                modified_blocks.add(modified_fluent[1])
            elif modified_fluent[1] in platforms:
                modified_platforms.add(modified_fluent[1])

    plot_pigs_and_birds(modified_birds, modified_pigs, pddl_state, ax)

    # PLot blocks
    for block in modified_blocks:
        plot_block(block, pddl_state, ax)

    # plot platforms
    for platform in modified_platforms:
        plot_platform(platform, pddl_state, ax)



''' Simulate the given action in the given state using the given meta model'''
def plot_expected_trace_for_obs(meta_model: MetaModel,
                                observation: ScienceBirdsObservation,
                                delta_t = 0.05,
                                ax=None):
    # Repair angle
    expected_trace = PddlPlusSimulator().simulate_observed_action(observation.state, observation.action, meta_model, delta_t)
    state_sequence = [timed_state[0] for timed_state in expected_trace]
    global BIRD_MARKER
    BIRD_MARKER = ".b"
    return plot_state_sequence(state_sequence, meta_model.create_pddl_state(observation.state), ax)

''' Plot the given observation'''
def plot_observation(observation: ScienceBirdsObservation, ax=None, marker="o"):
    meta_model = MetaModel()
    sb_state = observation.state
    # pddl_state = meta_model.create_pddl_state(sb_state)
    initial_state = PddlPlusState(meta_model.create_pddl_problem(sb_state).init)
    obs_state_sequence = observation.get_trace(meta_model)
    global BIRD_MARKER
    BIRD_MARKER = "xr"
    return plot_state_sequence(obs_state_sequence, initial_state, ax, marker)

''' Plots the expected vs observated trajectories '''
def plot_expected_vs_observed(meta_model: MetaModel, observation: ScienceBirdsObservation, fig = None):
    matplotlib.interactive(True)
    if fig==None:
        _, fig = plt.subplots()
    plot_expected_trace_for_obs(meta_model, observation, ax = fig)
    plot_observation(observation, ax=fig)
    return fig

'''
Plotting a PDDL+ state. '''
def plot_pddl_state(pddl_state : PddlPlusState, ax ):
    plot_pigs_and_birds(pddl_state.get_birds(), pddl_state.get_pigs(), pddl_state, ax)

    # PLot blocks
    blocks = pddl_state.get_blocks()
    for block in blocks:
        plot_block(block, pddl_state, ax)

    # plot platforms
    platforms = pddl_state.get_platforms()
    for platform in platforms:
        plot_platform(platform, pddl_state, ax)