import time

from matplotlib import pyplot as plt, patches as patches

from agent.consistency.fast_pddl_simulator import CachingPddlPlusSimulator
from agent.consistency.nyx_pddl_simulator import NyxPddlPlusSimulator
from agent.planning.pddl_plus import PddlPlusState
import matplotlib
import matplotlib.animation as animation
from agent.planning.meta_model import *
from agent.consistency.observation import *
from agent.consistency.pddl_plus_simulator import *
import matplotlib._color_data as mcd
from agent.planning.sb_meta_model import *
import numpy as np

from agent.planning.sb_meta_model import ScienceBirdsMetaModel

BIRD_MARKER = ".r"
PIG_MARKER = "*"

object_type_to_color = {
    "pig": mcd.TABLEAU_COLORS["tab:pink"],
    "bird": "r",
    "block": "gray",
    "platform": "black"
}
object_type_to_marker = {
    "pig": "o",
    "bird": "."
}


def plot_state_sequence(pddl_state_seq: list, pddl_state: PddlPlusState, ax=None, marker="x"):
    """
    Plotting a sequence of states, showing where the pigs, platforms, and birds are initially,
    and showing the trajectory of the active bird.
    """
    if ax is None:
        _, ax = plt.subplots()
    # plot_pddl_state(pddl_state, ax)

    # plot_intermediate_state(obs_state, previous_state, ax)

    first_state = plot_pddl_state(pddl_state_seq[0], ax)
    (left, right_x) = plt.xlim()
    (left, right_y) = plt.ylim()
    max_axis = max(right_x, right_y)
    plt.xlim((0, max_axis))
    plt.ylim((0, max_axis))

    plt.show()

    previous_state = pddl_state
    for intermediate_pddl_state in pddl_state_seq[1:]:
        # plot_intermediate_state(intermediate_pddl_state, previous_state, ax)
        plot_pddl_state(intermediate_pddl_state, ax)
        previous_state = intermediate_pddl_state
        plt.show()

    #
    #
    # obs_state = pddl_state_seq[0]
    #

    # Set plot area to be a square, so that proportions are right.

    return ax


def plot_platform(platform, pddl_state, ax):
    """ Plot a platform"""
    x = pddl_state[("x_platform", platform)]
    y = pddl_state[("y_platform", platform)]
    width = pddl_state[("platform_width", platform)]
    height = pddl_state[("platform_height", platform)]
    leftmost_x = x - width / 2
    lowermost_y = y - height / 2
    rect = patches.Rectangle((leftmost_x, lowermost_y), width, height, linewidth=1, edgecolor='black',
                             facecolor=object_type_to_color["platform"])
    ax.add_patch(rect)


def plot_block(block, pddl_state, ax):
    """ Plot a block """
    x = pddl_state[("x_block", block)]
    y = pddl_state[("y_block", block)]
    width = pddl_state[("block_width", block)]
    height = pddl_state[("block_height", block)]
    leftmost_x = x - width / 2
    lowermost_y = y - height / 2
    rect = patches.Rectangle((leftmost_x, lowermost_y),
                             width, height,
                             linewidth=1,
                             edgecolor='black',
                             facecolor=object_type_to_color["block"])
    ax.add_patch(rect)
    ax.annotate(block[:1], (x, y), color='w', weight='bold', fontsize=6, ha='center', va='center')


def plot_pigs_and_birds(modified_birds, modified_pigs, pddl_state: PddlPlusState, ax):
    """ Plot the pigs and birds in the pddl state """
    # plot birds
    x_birds = [pddl_state[("x_bird", bird)] for bird in modified_birds]
    y_birds = [pddl_state[("y_bird", bird)] for bird in modified_birds]
    ax.plot(x_birds, y_birds,
            object_type_to_marker["bird"],
            color=object_type_to_color["bird"],
            linestyle="")

    x_pigs = [pddl_state[("x_pig", pig)] for pig in modified_pigs]
    y_pigs = [pddl_state[("y_pig", pig)] for pig in modified_pigs]
    ax.plot(x_pigs, y_pigs,
            object_type_to_marker["pig"],
            color=object_type_to_color["pig"],
            linestyle="")


def plot_expected_trace_for_obs(meta_model: MetaModel,
                                observation: ScienceBirdsObservation,
                                delta_t=0.05,
                                ax=None):
    """ Simulate the given action in the given state using the given meta model"""
    # Repair angle
    expected_trace = NyxPddlPlusSimulator().simulate_observed_action(observation.state, observation.action, meta_model,
                                                                     delta_t)
    state_sequence = [timed_state[0] for timed_state in expected_trace]
    global BIRD_MARKER
    BIRD_MARKER = ".b"
    return plot_state_sequence(state_sequence, meta_model.create_pddl_state(observation.state), ax)


def plot_sb_observation(observation: ScienceBirdsObservation, ax=None, marker="o"):
    """ Plot the given observation"""
    meta_model = ScienceBirdsMetaModel()
    sb_state = observation.state
    # pddl_state = meta_model.create_pddl_state(sb_state)
    initial_state = PddlPlusState(meta_model.create_pddl_problem(sb_state).init)
    obs_state_sequence = observation.get_pddl_states_in_trace(meta_model)
    return plot_state_sequence(obs_state_sequence, initial_state, ax, marker)


def plot_sb_initial_state(observation: ScienceBirdsObservation, ax=None):
    """ Plots the initial state of the given observation"""
    meta_model = ScienceBirdsMetaModel()
    sb_state = observation.state
    initial_state = PddlPlusState(meta_model.create_pddl_problem(sb_state).init)
    return plot_pddl_state(initial_state, ax)


def plot_expected_vs_observed(meta_model: MetaModel, observation: ScienceBirdsObservation, fig=None):
    """ Plots the expected vs observated trajectories """
    matplotlib.interactive(True)
    if fig == None:
        _, fig = plt.subplots()
    plot_expected_trace_for_obs(meta_model, observation, ax=fig)
    plot_sb_observation(observation, ax=fig)
    return fig


def plot_pddl_state(pddl_state: PddlPlusState, ax=None):
    """
    Plotting a PDDL+ state. """
    if ax is None:
        _, ax = plt.subplots()

    plot_pigs_and_birds(pddl_state.get_birds(), pddl_state.get_pigs(), pddl_state, ax)

    # PLot blocks
    blocks = pddl_state.get_blocks()
    for block in blocks:
        plot_block(block, pddl_state, ax)

    # plot platforms
    platforms = pddl_state.get_platforms()
    for platform in platforms:
        plot_platform(platform, pddl_state, ax)

    return ax


def animate_trace(fig, ax, obs_state_sequence: list, interval=100):
    """ Create an animation that visalizes the states on the trace
    fig and ax are the figure and subplot on which to plot"""

    def animate(i):
        ax.clear()
        plot_pddl_state(obs_state_sequence[i + 1], ax)
        return ax

    return animation.FuncAnimation(fig, animate, frames=np.arange(len(obs_state_sequence) - 1), interval=interval,
                                   repeat=False)


def plot_intermediate_state(pddl_state: PddlPlusState, previous_pddl_state: PddlPlusState, ax):
    """ Plotting an intermediate PDDL+ state. Do not plot an object if it did not change """
    modified_fluents = set()
    for numeric_fluent in pddl_state.numeric_fluents:
        if numeric_fluent in previous_pddl_state.numeric_fluents and \
                pddl_state[numeric_fluent] == previous_pddl_state[numeric_fluent]:
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


def animate_expected(output_path, sb_ob, meta_model=ScienceBirdsMetaModel()):
    """ Create an animated gif file showing the expected trace for observed state and action """
    simulator = CachingPddlPlusSimulator()
    sim_trace = simulator.simulate_observed_action(sb_ob.state, sb_ob.action, meta_model)
    sim_state_sequence = [timed_state[0] for timed_state in sim_trace]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    trace_animation = animate_trace(fig, ax, sim_state_sequence)
    trace_animation.save(output_path)


def animate_observed(output_path, sb_ob, meta_model=ScienceBirdsMetaModel()):
    """ Create an animated gif file showing the observed trace """
    obs_state_sequence = sb_ob.get_pddl_states_in_trace(meta_model)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    trace_animation = animate_trace(fig, ax, obs_state_sequence)
    trace_animation.save(output_path)
