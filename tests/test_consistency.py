import matplotlib.pyplot as plt
import pytest

from agent.repair.sb_repair import BirdLocationConsistencyEstimator
from agent.sb_hydra_agent import *
from agent.planning.model_manipulator import ManipulateInitNumericFluent
from agent.planning.sb_planner import *

from agent.perception.perception import *
from tests.test_utils import create_logger
from os import listdir
from agent.consistency import trace_visualizer as trace_visualizer
logger = create_logger("test_consistency")

DATA_DIR = path.join(settings.ROOT_PATH, 'data')
TEST_DATA_DIR = path.join(DATA_DIR, 'science_birds', 'tests')
TEMP_TEST_DATA_DIR = path.join(TEST_DATA_DIR, 'temp')
PROBLEM_FILE = path.join(DATA_DIR, "sb_prob_l1.pddl")
DOMAIN_FILE = path.join(DATA_DIR, "sb_domain_l1.pddl")
PLAN_FILE = path.join(DATA_DIR, "docker_plan_trace_l1.txt")

PRECISION = 1
Y_BIRD_FLUENT = ('y_bird', 'redbird_0')
X_BIRD_FLUENT = ('x_bird', 'redbird_0')
DELTA_T = 0.05
GRAVITY_CHANGE = -50
GRAVITY = "gravity"

''' Helper function: loads plan, problem, and domain used to evaluate consistency checker'''
def _load_plan_problem_domain():
    (problem, domain) = test_utils.load_problem_and_domain(PROBLEM_FILE,DOMAIN_FILE)
    plan = SBPlanner().extract_actions_from_plan_trace(PLAN_FILE)
    return (plan, problem, domain)

def _print_fluents_values(state_seq:list, fluent_names:list):
    headers = ",".join([str(fluent_name) for fluent_name in fluent_names])
    print("\n%s" % headers)
    for state in state_seq:
        line = ",".join(["%.2f" % state[fluent_name] for fluent_name in fluent_names])
        print("%s" % line)

#
# class TimeIndependentConsistencyEstimator(ConsistencyEstimator):
#     """
#     Computes inconsistency of a set of fluents, where the timing information of the simulated and observed traces might
#     not match.
#     """
#
#     def __init__(self, fluent_names, obs_prefix=100, discount_factor=0.25, consistency_threshold=20):
#         """
#         Specify which fluents to check, and the size of the observed sequence prefix to consider.
#         This is because we acknowledge that later in the observations, our model is less accurate.
#         """
#         super().__init__(fluent_names, obs_prefix, discount_factor, consistency_threshold)
#
#     def consistency_from_trace(self, simulation_trace: list, state_seq: list, delta_t: float = DEFAULT_DELTA_T):
#         """ Returns a value indicating the estimated consistency. """
#         # Only consider states with some info regarding the relevant fluents
#         states_with_info = []
#         for state in state_seq:
#             has_info = False
#             for fluent_name in self.fluent_names:
#                 if fluent_name in state:
#                     has_info = True
#                     break
#             if has_info:
#                 states_with_info.append(state)
#         state_seq = states_with_info
#
#         # Compute consistency of every observed state
#         consistency_per_state = self._compute_consistency_per_state(simulation_trace, state_seq, delta_t)
#
#         # Aggregate the consistency
#         discount = 1.0
#         max_error = 0
#         for i, consistency in enumerate(consistency_per_state):
#             if i > self.obs_prefix:
#                 break
#             if consistency > self.consistency_threshold:
#                 weighted_error = consistency * discount
#                 if max_error < weighted_error:
#                     max_error = weighted_error
#
#             discount = discount * self.discount_factor
#
#         return max_error
#
#     def _compute_consistency_per_state(self, expected_state_seq: list, observed_states: list,
#                                        delta_t: float = DEFAULT_DELTA_T):
#         """
#         Returns a vector of values, one per observed state, indicating how much it is consistent with the simulation.
#         The first parameter is a list of (state,time) pairs, the second is just a list of states
#         Current implementation ignores order, and just looks for the best time for each state in the state_seq,
#         and ignore cases where the fluent is not in the un-timed state seqq aiming to minimize its distance from the fitted
#         piecewise-linear interpolation.
#         """
#         t_values, fluent_to_expected_values = self._fit_expected_values(expected_state_seq, delta_t=delta_t)
#         consistency_per_state = []
#         for i, state in enumerate(observed_states):
#             (best_fit_t, min_error) = self._compute_best_fit(state, t_values, fluent_to_expected_values)
#             consistency_per_state.append(min_error)
#             if i >= self.obs_prefix:  # We only consider a limited prefix of the observed sequence of states
#                 break
#         return consistency_per_state
#
#     def _fit_expected_values(self, simulation_trace, delta_t=0.01):
#         """
#         Create a piecewise linear interpolation for the given timed_state_seq
#         """
#         # Get values over time for each fluent
#         fluent_to_values = dict()
#         fluent_to_times = dict()
#         for (state, t, _) in simulation_trace:
#             for fluent_name in self.fluent_names:
#                 if fluent_name in state.numeric_fluents:
#                     fluent_value = state[fluent_name]
#                     if fluent_name not in fluent_to_times:
#                         fluent_to_times[fluent_name] = []
#                         fluent_to_values[fluent_name] = []
#                     fluent_to_times[fluent_name].append(t)
#                     fluent_to_values[fluent_name].append(float(fluent_value))
#
#                     # Fit a piecewise linear function to each fluent
#         max_t = max([fluent_to_times[fluent_name][-1] for fluent_name in fluent_to_times])
#         fluent_to_expected_values = dict()
#         all_t_values = np.arange(0, max_t, delta_t)
#         for fluent_name in self.fluent_names:
#             t_values = fluent_to_times[fluent_name]
#             fluent_values = fluent_to_values[fluent_name]
#             fitted_values = np.interp(all_t_values, t_values, fluent_values)
#             fluent_to_expected_values[fluent_name] = fitted_values
#         return all_t_values, fluent_to_expected_values
#
#     def _compute_best_fit(self, state, t_values, fluent_to_expected_values):
#         """ Compute the t value that best fits the given state. Returns
#             this t and the error at that time, i.e., the difference between
#             the state's fluent values and their expected values at the best fit time. """
#         best_fit_error = float('inf')
#         best_t = -1
#         for t in range(len(t_values)):
#             error_at_t = 0
#             for fluent_name in self.fluent_names:
#                 if fluent_name not in state:  # TODO: A design choice. Ignore missing fluent values
#                     continue
#                 fluent_value = float(state[fluent_name])
#                 consistent_fluent_value = fluent_to_expected_values[fluent_name][t]
#                 fluent_error = abs(fluent_value - consistent_fluent_value)
#                 if fluent_error > error_at_t:
#                     error_at_t = fluent_error
#             if error_at_t < best_fit_error:
#                 best_fit_error = error_at_t
#                 best_t = t
#         return best_t, best_fit_error
#
# ''' Test the single numeric fluent consistency checker, for the case where the numeric fluent is not constant,
#  and the outcome of the tests should be that the sequences are inconsistent. '''
# def test_multiple_numeric_fluent_inconsistent():
#     # Get expected timed state sequence according to the model (plan, problem, domain)
#     (plan, problem, domain) = _load_plan_problem_domain()
#     timed_state_seq = test_utils.simulate_plan_trace(plan, problem, domain, DEFAULT_DELTA_T)
#
#     # Modify the model
#     manipulator = ManipulateInitNumericFluent([GRAVITY],GRAVITY_CHANGE)
#     manipulator.apply_change(domain, problem)
#
#     # Get the new expected timed state sequence, according to the modified model
#     simulation_trace = test_utils.simulate_plan_trace(plan, problem, domain, DEFAULT_DELTA_T)
#     # Assert new timed state sequence is different from the original timed state sequence
#     diff_list = diff_traces(timed_state_seq, simulation_trace)
#     logging.info("\n".join(diff_list))
#     assert len(diff_list)>0
#
#     # Create a un-timed state sequence, to simulate how the SB observations would look like.
#     modified_state_seq = [state for (state, t, _) in simulation_trace]
#
#     # Assert the un-timed sequence created by the modified model is inconsistent with the timed sequence created by the original model
#     consistency_checker = TimeIndependentConsistencyEstimator([X_BIRD_FLUENT, Y_BIRD_FLUENT],
#                                                               obs_prefix=float('inf'),
#                                                               discount_factor=1.0)
#     consistent_score = consistency_checker.consistency_from_trace(timed_state_seq, modified_state_seq)
#     assert consistent_score > PRECISION


''' Test the single numeric fluent consistency checker, for the case where the numeric fluent is not constant,
 and the outcome of the tests should be that the sequences are inconsistent. '''
def test_all_bird_numeric_fluent_inconsistent():
    # Get expected timed state sequence according to the model (plan, problem, domain)
    (plan, problem, domain) = _load_plan_problem_domain()
    simulation_trace = test_utils.simulate_plan_trace(plan, problem, domain, DEFAULT_DELTA_T)

    consistency_checker = BirdLocationConsistencyEstimator(discount_factor=1.0, unique_prefix_size=float('inf'))
    # Assert the un-timed sequence created by the original model is consistent with the timed sequence created by the original model
    original_state_seq = [state for (state,t,_) in simulation_trace]
    consistency_score = consistency_checker.consistency_from_trace(simulation_trace, original_state_seq)
    assert consistency_score < PRECISION


    # Modify the model
    manipulator = ManipulateInitNumericFluent([GRAVITY],GRAVITY_CHANGE)
    manipulator.apply_change(domain, problem)

    # Get the new expected timed state sequence, according to the modified model
    modified_simulation_trace =test_utils.simulate_plan_trace(plan, problem, domain, DEFAULT_DELTA_T)

    # Assert new timed state sequence is different from the original timed state sequence
    diff_list = diff_traces(simulation_trace, modified_simulation_trace)
    logging.info("\n".join(diff_list))
    assert len(diff_list)>0

    # Create a un-timed state sequence, to simulate how the SB observations would look like.
    modified_state_seq = [state for (state, t,_) in modified_simulation_trace]
    # Assert the un-timed sequence created by the modified model is inconsistent with the timed sequence created by the original model
    consistent_score = consistency_checker.consistency_from_trace(simulation_trace, modified_state_seq)
    assert consistent_score > PRECISION

    consistent_score = consistency_checker.consistency_from_trace(simulation_trace, modified_state_seq)
    assert consistent_score > PRECISION

#################### System Test ###############################
@pytest.fixture(scope="module")
def launch_science_birds():
    print("starting")
    env = sb.ScienceBirds(None,launch=True,config='test_consistency_config.xml')
    yield env
    print("teardown tests")
    env.kill()

''' Run Hydra, collect observations, check for consistency '''
@pytest.mark.skip("Have not migrated to 0.3.6")
def test_consistency_in_agent(launch_science_birds):
    settings.SB_SIM_SPEED = 5
    settings.SB_GT_FREQ = 1
    save_obs = True

    # Launch SB and play a level
    env = launch_science_birds
    hydra = SBHydraAgent(env)
    hydra.run_next_action()
    our_observation = hydra.find_last_obs()

    # Save the observation file for debug and for the offline test
    if save_obs:
        obs_output_file = path.join(TEST_DATA_DIR, "obs_test_consistency_in_agent.p")
    else:
        obs_output_file = path.join(TEMP_TEST_DATA_DIR, "obs_test_consistency_in_agent.p")

    pickle.dump(our_observation, open(obs_output_file, "wb")) #*** uncomment if needed for debugging ***

    plt.interactive(True)
    _, fig = plt.subplots()
    test_utils.plot_observation(our_observation, ax=fig)
    test_utils.plot_expected_trace_for_obs(hydra.meta_model, our_observation, ax=fig)

    # Check consistent with correct model
    consistency_estimator = BirdLocationConsistencyEstimator()
    good_consistency = check_obs_consistency(our_observation,hydra.meta_model, consistency_estimator)

    # Check consistent with incorrect model
    gravity_factor = hydra.meta_model.constant_numeric_fluents["gravity_factor"]
    bad_meta_model = ScienceBirdsMetaModel()
    bad_meta_model.constant_numeric_fluents["gravity_factor"] = gravity_factor/2
    bad_consistency = check_obs_consistency(our_observation, bad_meta_model, consistency_estimator)
    test_utils.plot_observation(our_observation, ax=fig)
    test_utils.plot_expected_trace_for_obs(bad_meta_model, our_observation, ax=fig)

    assert good_consistency < bad_consistency


''' Run Hydra, collect observations, check for consistency '''
# @pytest.mark.skipif(True, reason="Modified planner fails on basic levels")
@pytest.mark.skip("Old observation file needs to be updated to 0.3.6")
def test_consistency_in_agent_offline():
    plot_exp_vs_obs = False

    obs_output_file = path.join(TEST_DATA_DIR, "obs_test_consistency_in_agent.p")
    our_observation = pickle.load(open(obs_output_file, "rb"))
    meta_model = ScienceBirdsMetaModel()

    # Uncomment for debug:
    # plt.interactive(True)
    # _, fig = plt.subplots()
    # fig = test_utils.plot_observation(our_observation, ax=fig)
    # fig = test_utils.plot_expected_trace_for_obs(meta_model, our_observation, ax=fig)

    # Check consistent with correct model
    consistency_estimator = BirdLocationConsistencyEstimator()
    good_consistency = check_obs_consistency(our_observation, meta_model, consistency_estimator, plot_obs_vs_exp=plot_exp_vs_obs)

    # Check consistent with incorrect model
    bad_meta_model = ScienceBirdsMetaModel()
    gravity_factor = meta_model.constant_numeric_fluents["gravity_factor"]
    bad_meta_model.constant_numeric_fluents["gravity_factor"] = gravity_factor/2
    bad_consistency = check_obs_consistency(our_observation, bad_meta_model, consistency_estimator, plot_obs_vs_exp=plot_exp_vs_obs)

    assert good_consistency < bad_consistency

''' Run Hydra, collect observations, check for consistency '''
@pytest.mark.skipif(True, reason="Have not migrated yet")
def test_bad_shot_consistency(launch_science_birds):
    settings.SB_SIM_SPEED = 5
    settings.SB_GT_FREQ = 1
    plot_me = False
    save_obs = True

    env = launch_science_birds
    hydra = SBHydraAgent(env)
    angle = 75
    hydra.planner = PlannerStub(angle)
    hydra.run_next_action()
    meta_model = hydra.meta_model
    our_observation = hydra.find_last_obs()

    # Save the observation file for debug and for the offline test
    if save_obs:
        obs_output_file = path.join(TEST_DATA_DIR, "test_bad_shot_consistency_obs.p")
    else:
        obs_output_file = path.join(TEMP_TEST_DATA_DIR, "test_bad_shot_consistency_obs.p")
    pickle.dump(our_observation, open(obs_output_file, "wb"))  # *** uncomment if needed for debugging ***

    # For debug purposes
    # plt.interactive(True)
    # _, fig = plt.subplots()
    # fig = test_utils.plot_observation(our_observation, ax = fig)
    # fig = test_utils.plot_expected_trace_for_obs(hydra.meta_model, our_observation, ax=fig)

    # Check consistent with correct model
    consistency_estimator = BirdLocationConsistencyEstimator()
    consistency = check_obs_consistency(our_observation,meta_model, consistency_estimator, plot_obs_vs_exp=plot_me)

    # Check consistent with incorrect model
    good_gravity_factor = meta_model.constant_numeric_fluents["gravity_factor"]

    bad_meta_model = ScienceBirdsMetaModel()
    bad_meta_model.constant_numeric_fluents["gravity_factor"] = good_gravity_factor/2
    bad_consistency = check_obs_consistency(our_observation, bad_meta_model, consistency_estimator, plot_obs_vs_exp=plot_me)

    assert bad_consistency>consistency


# Constants for ScienceBirds
SB_NON_NOVEL_OBS_DIR = path.join(settings.ROOT_PATH, 'data', 'science_birds', 'consistency', 'non_novel')
SB_NOVEL_OBS_DIR = path.join(settings.ROOT_PATH, 'data', 'science_birds', 'consistency', 'novel')

SB_NON_NOVEL_TESTS = listdir(SB_NON_NOVEL_OBS_DIR)
SB_NOVEL_TESTS = listdir(SB_NOVEL_OBS_DIR)



from agent.consistency.sequence_consistency_estimator import *
# ''' Checks consistency by considering the location of the Cartpole fluents '''
# class ScienceBirdsSequenceConsistencyEstimator(MetaModelBasedConsistencyEstimator):
#     def __init__(self, unique_prefix_size = 100,discount_factor=0.9, consistency_threshold = 0.01):
#         self.unique_prefix_size=unique_prefix_size
#         self.discount_factor = discount_factor
#         self.consistency_threshold = consistency_threshold
#
#     ''' Estimate consitency by considering the location of the birds in the observed state seq '''
#     def consistency_from_trace(self, simulation_trace: list, state_seq: list, delta_t: float = DEFAULT_DELTA_T):
#         fluent_names = []
#         fluent_names.append(('x',))
#         fluent_names.append(('x_dot',))
#         fluent_names.append(('theta',))
#         fluent_names.append(('theta_dot',))
#
#         consistency_checker = SequenceConsistencyEstimator(fluent_names, self.unique_prefix_size, self.discount_factor, self.consistency_threshold)
#         return consistency_checker.consistency_from_trace(simulation_trace, state_seq, delta_t)



def test_consistency_sb_false_positives():
    ''' Drill down on a single observation that outputs a false positive for our consistency checker '''
    meta_model = ScienceBirdsMetaModel()
    samples = 10

    # consistency_checker = ScienceBirdsConsistencyEstimator()
    consistency_checker = ScienceBirdsConsistencyEstimator()

    for i, ob_file in enumerate(SB_NON_NOVEL_TESTS[3:samples]):
        # load file
        sb_ob = pickle.load(open(path.join(SB_NON_NOVEL_OBS_DIR, ob_file), "rb"))

        matplotlib.interactive(True)
        plot_axes = trace_visualizer.plot_sb_observation(sb_ob)
        trace_visualizer.plot_expected_trace_for_obs(meta_model, sb_ob, ax=plot_axes)


        consistency = check_obs_consistency(sb_ob, meta_model, consistency_checker)
        logger.info("Non-novel file {}: {}, consistency {}".format(i,ob_file, consistency))
        assert consistency<settings.SB_CONSISTENCY_THRESHOLD


def test_consistency_sb():
    ''' Drill down on a single observation that outputs a false positive for our consistency checker '''
    meta_model = ScienceBirdsMetaModel()

    samples = 10
    true_negatives = 0
    true_positives = 0
    false_negatives = 0
    false_positives = 0

    print("Non novel cases:")
    consistency_checker = ScienceBirdsConsistencyEstimator()

    for ob_file in SB_NON_NOVEL_TESTS[:samples]:
        # load file
        sb_ob = pickle.load(open(path.join(SB_NON_NOVEL_OBS_DIR, ob_file), "rb"))
        consistency = check_obs_consistency(sb_ob, meta_model, consistency_checker)
        print("Non-novel file {}, consistency {}".format(ob_file, consistency))
        assert consistency<settings.SB_CONSISTENCY_THRESHOLD

    print("Novel cases:")
    for ob_file in SB_NOVEL_TESTS[:samples]:
        sb_ob = pickle.load(open(path.join(SB_NOVEL_OBS_DIR, ob_file), "rb"))
        consistency = check_obs_consistency(sb_ob, meta_model, consistency_checker)
        print("Novel file {}, consistency {}".format(ob_file, consistency))
        assert consistency >= settings.SB_CONSISTENCY_THRESHOLD
