from agent.consistency.trace_visualizer import plot_expected_trace_for_obs
from agent.reward_estimation.reward_estimator import RewardEstimator
import pickle
import datetime
import time

import pandas

import settings
from agent.hydra_agent import logger, NN_PROB, PDDL_PROB, NOVELTY_EXISTENCE_NOT_GIVEN, NOVELTY_LIKELIHOOD
from agent.planning.sb_planner import SBPlanner
from agent.repair.meta_model_repair import *
import numpy
# from state_prediction.anomaly_detector_fc_multichannel import FocusedSBAnomalyDetector

# TODO: Maybe push this to the settings file? then every module just adds a logger
from agent.repair.sb_repair import ScienceBirdsConsistencyEstimator, ScienceBirdsMetaModelRepair
from agent.gym_hydra_agent import REPAIR_CALLS, REPAIR_TIME, logger
from utils.point2D import Point2D
from worlds.science_birds_interface.client.agent_client import GameState
from agent.hydra_agent import HydraAgent

logging.basicConfig(format='%(name)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("hydra_agent")

# Flags from ANU
# NOVELTY_EXISTENCE_NOT_GIVEN = -1
# The self.novelty_existence value indicating that novelty detection is not given by the environment

# stats_per_level dictionary keys
# NN_PROB = "nn_novelty_likelihood" this originally was the state-based detector written by UPenn
REWARD_PROB = "reward_estimator_likelihood"
PDDL_PROB = "pddl_novelty_likelihood"
NOVELTY_LIKELIHOOD = "novelty_likelihood"
UNKNOWN_OBJ = "unknown_object"
PLAN = "plan"
UNDEFINED = None

ENSEMBLE_MODEL = "{}/model/ensemble_simple_22_25.pkl".format(settings.ROOT_PATH)


class SBHydraAgent(HydraAgent):
    """
    Probably needs to subclass for each domain. We will cross that bridge when we get there
    """

    def __init__(self, env=None, agent_stats=None):
        if agent_stats is None:
            agent_stats = list()
        logger.info("[hydra_agent_server] :: Agent Created")

        super().__init__(planner=SBPlanner(ScienceBirdsMetaModel()),
                         meta_model_repair=None)

        # Default values
        self.env = env  # agent always has a pointer to its environment
        if env is not None:
            env.sb_client.set_game_simulation_speed(settings.SB_SIM_SPEED)
        self.agent_stats = agent_stats
        self.consistency_estimator = ScienceBirdsConsistencyEstimator()
        self.current_level = 0
        self.novelty_detections = list()
        self.initialize_processing_state_variables()
        self._new_novelty_likelihood = settings.NOVELTY_POSSIBLE

        self.reward_estimator = RewardEstimator()

        # state processing variables:
        self.perception = Perception()
        self.completed_levels = []
        self.observations = []

        self.novel_objects = []

        self.cumulative_plan_time = 0.0
        self.overall_plan_time = 0.0
        self.shot_num = 0
        self.trial_timestamp = datetime.datetime.now().strftime("%y%m%d%H%M%S")

        self.stats_for_level = dict()
        self.level_novelty_indicators = {
            REWARD_PROB: list(),
            PDDL_PROB: list(),
            UNKNOWN_OBJ: list(),
            PLAN: list()
        }

        self.nn_prob_per_level = []
        self.pddl_prob_per_level = []
        self.num_objects = 0

        # record latest angle for repair module
        self.latest_angle = None

        # fields collected from random functions:
        self.training_level_backup = 0

        # Simple repair tracker
        self._need_to_repair = False
        self.made_plan = False

    def initialize_processing_state_variables(self):
        self.perception = Perception()
        self.completed_levels = []
        self.observations = []

        self.novel_objects = []

        self.cumulative_plan_time = 0.0
        self.overall_plan_time = 0.0
        self.shot_num = 0
        self.trial_timestamp = datetime.datetime.now().strftime("%y%m%d%H%M%S")

        self.stats_for_level = dict()
        self.level_novelty_indicators = {
            REWARD_PROB: list(),
            PDDL_PROB: list(),
            UNKNOWN_OBJ: list(),
            PLAN: list()
        }

        self.novelty_detections = list()
        self.nn_prob_per_level = []
        self.pddl_prob_per_level = []
        self.num_objects = 0
        self._new_novelty_likelihood = False

    def reinit(self):
        """ Prepare this agent for a new trial. """
        logging.info('Reinit...')
        self.env.history = []
        self.meta_model = ScienceBirdsMetaModel()
        self.planner = SBPlanner(self.meta_model)  # TODO: Discuss this w. Wiktor & Matt
        self.initialize_processing_state_variables()

    def main_loop(self, max_actions=1000):
        """ Runs the agent. Returns False if the evaluation has not ended, and True if it has ended."""

        logger.info("[hydra_agent_server] :: Entering main loop")
        logger.info("[hydra_agent_server] :: Delta t = {}".format(str(settings.SB_DELTA_T)))
        logger.info("[hydra_agent_server] :: Simulation speed = {}\n\n".format(str(settings.SB_SIM_SPEED)))
        logger.info("[hydra_agent_server] :: Planner memory limit = {}".format(str(settings.SB_PLANNER_MEMORY_LIMIT)))
        logger.info("[hydra_agent_server] :: Planner timeout = {}s\n\n".format(str(settings.SB_TIMEOUT)))
        t = 0

        self.overall_plan_time = time.perf_counter()
        self.cumulative_plan_time = 0

        while True:

            observation = ScienceBirdsObservation()  # Create an observation object to track on what happend
            raw_state = self.env.get_current_state()

            if raw_state.game_state.value == GameState.PLAYING.value:
                self.env.sb_client.batch_ground_truth(10, 1)
                raw_state = self.env.get_current_state()
                self.handle_game_playing(observation, raw_state)
                if settings.NOVELTY_POSSIBLE:
                    self.num_objects = len(raw_state.objects[0]['features'])
                    # print("number of objects is {}".format(self.num_objects))
                    self._record_novelty_indicators(observation)
            elif raw_state.game_state.value == GameState.WON.value:
                self.handle_game_won()
            elif raw_state.game_state.value == GameState.LOST.value:
                self.handle_game_lost()
            elif raw_state.game_state.value == GameState.NEWTRAININGSET.value:
                self.handle_new_training_set()
            elif raw_state.game_state.value == GameState.EVALUATION_TERMINATED.value:
                return self.handle_evaluation_terminated()
            elif raw_state.game_state.value == GameState.REQUESTNOVELTYLIKELIHOOD.value:
                self.handle_request_novelty_likelihood()
                time.sleep(5 / settings.SB_SIM_SPEED)
            elif raw_state.game_state.value == GameState.NEWTRIAL.value:
                self.handle_new_trial()
            elif raw_state.game_state.value == GameState.MAIN_MENU.value:
                self.handle_main_menu()
            else:
                logger.info(
                    "[hydra_agent_server] :: Unexpected state.game_state.value {}".format(raw_state.game_state.value))
                assert False
            t += 1

            self.observations.append(observation)

    def handle_main_menu(self):
        logger.info("unexpected main menu page, reload the level : %s" % self.current_level)
        self.current_level = self.env.sb_client.load_next_available_level()

    #        self.novelty_existence = self.env.sb_client.get_novelty_info()

    def handle_new_trial(self):
        """ Handle what happens when the agent receives a NEWTRIAL request"""

        # DO something to start a fresh agent for a new training set
        (time_limit, interaction_limit, n_levels, attempts_per_level, mode, seq_or_set,
         allowNoveltyInfo) = self.env.sb_client.ready_for_new_set()
        logger.info("New Trial Request Received. Refresh agent.")
        self.reinit()
        self.current_level = 0
        self.training_level_backup = 0
        change_from_training = True
        self.current_level = self.env.sb_client.load_next_available_level()
        # self.novelty_existence = self.env.sb_client.get_novelty_info()

    def handle_request_novelty_likelihood(self):
        """ Handle what happens when the agent receives a REQUESTNOVELTYLIKELIHOOD request"""

        logger.info("[hydra_agent_server] :: Requesting Novelty Likelihood. Novelty likelihood is {}".format(
            self._new_novelty_likelihood))
        if self._new_novelty_likelihood:
            novelty_likelihood = 1
        else:
            novelty_likelihood = 0

        non_novelty_likelihood = 1 - novelty_likelihood

        # placeholders for novelty information
        if len(self.novel_objects) > 0:
            ids = set([int(object_id_str) for object_id_str in self.novel_objects])
            novelty_description = "Unknown type of objects detected"
        else:
            ids = set()
            novelty_description = "Uncharacterized novelty"
        novelty_level = 0

        logger.info("[hydra_agent_server] :: Reporting novelty_likelihood: {}".format(novelty_likelihood))
        self.env.sb_client.report_novelty_likelihood(novelty_likelihood, non_novelty_likelihood, ids, novelty_level,
                                                     novelty_description)

    def handle_evaluation_terminated(self):
        """ Handle what happens when the agent receives a EVALUATION_TERMINATED request"""

        # store info and disconnect the agent as the evaluation is finished
        self._handle_end_of_level(True)
        logger.info("Evaluation complete.")
        return True

    def handle_new_training_set(self):
        """ Handle what happens when the agent receives a NEWTRAININGSET request"""

        # DO something to start a fresh agent for a new training set
        (time_limit, interaction_limit, n_levels, attempts_per_level, mode, seq_or_set,
         allowNoveltyInfo) = self.env.sb_client.ready_for_new_set()
        self.current_level = 0
        self.training_level_backup = 0
        change_from_training = True
        self.current_level = self.env.sb_client.load_next_available_level()
        # self.novelty_existence = self.env.sb_client.get_novelty_info()

    def handle_game_lost(self):
        """ Handle what happens when the agent receives a LOST request"""
        self._handle_end_of_level(False)

    def handle_game_won(self):
        """ Handle what happens when the agent receives a WON request"""
        self._handle_end_of_level(True)
        return self.cumulative_plan_time, self.overall_plan_time

    def _record_novelty_indicators(self, observation: ScienceBirdsObservation):
        logging.info("Computing novelty likelihood...")

        if self.novelty_existence in [0, 1]:
            self.level_novelty_indicators[PDDL_PROB].append(UNDEFINED)
            self.level_novelty_indicators[UNKNOWN_OBJ].append(UNDEFINED)
            self.level_novelty_indicators[REWARD_PROB].append(UNDEFINED)
            return

        if observation.hasUnknownObj():
            # self.level_novelty_indicators[PDDL_PROB].append(UNDEFINED)
            self.level_novelty_indicators[PDDL_PROB].append(
                1000)  ### add a high value because if there is a new object, the PDDL is highly inconsistent.
            self.level_novelty_indicators[UNKNOWN_OBJ].append(True)
            self.novel_objects = observation.get_novel_object_ids()
        else:
            self.level_novelty_indicators[UNKNOWN_OBJ].append(False)
            if settings.NO_PDDL_CONSISTENCY:
                pddl_prob = UNDEFINED
            else:
                pddl_prob = check_obs_consistency(observation, self.meta_model, self.consistency_estimator)
            self.level_novelty_indicators[PDDL_PROB].append(pddl_prob)

        difference = self.reward_estimator.compute_estimated_reward_difference(observation)
        self.level_novelty_indicators[REWARD_PROB].append(difference)

    def _detect_level_novelty(self):
        is_novel = False
        has_new_object = False

        if True in self.level_novelty_indicators[UNKNOWN_OBJ]:
            has_new_object = True

        pddl_consistency_list = [x for x in self.level_novelty_indicators[PDDL_PROB] if x is not None]
        if len(pddl_consistency_list) > 0:
            mean_pddl_inconsistency = sum(pddl_consistency_list) / len(pddl_consistency_list)
        else:
            mean_pddl_inconsistency = None

        if mean_pddl_inconsistency:
            are_level_observations_divergent = (mean_pddl_inconsistency > settings.SB_CONSISTENCY_THRESHOLD)
        else:
            are_level_observations_divergent = False

        reward_consistency_list = [x for x in self.level_novelty_indicators[REWARD_PROB] if x is not None]
        if len(reward_consistency_list) > 0:
            mean_reward_inconsistency = sum(reward_consistency_list) / len(reward_consistency_list)
        else:
            mean_reward_inconsistency = None

        if mean_reward_inconsistency:
            is_level_reward_inconsistent = (mean_reward_inconsistency > settings.SB_REWARD_CONSISTENCY_THRESHOLD)
        else:
            is_level_reward_inconsistent = False

        is_novel = has_new_object or are_level_observations_divergent or is_level_reward_inconsistent
        return is_novel

    def _detect_level_novelty_with_ensemble(self):

        with open(ENSEMBLE_MODEL, 'rb') as f:
            rf = pickle.load(f)

        if True in self.level_novelty_indicators[UNKNOWN_OBJ]:
            has_unknown_object = 1
        else:
            has_unknown_object = 0

        pddl_list = self.level_novelty_indicators[PDDL_PROB]

        if all(v is None for v in pddl_list):
            max_pddl_inconsistency = 1000
            avg_pddl_inconsistency = 1000
        else:
            max_pddl_inconsistency = numpy.nanmax(pddl_list)
            avg_pddl_inconsistency = numpy.nanmean(pddl_list)

        if len(self.level_novelty_indicators[REWARD_PROB]) == 0:
            max_reward_difference = 0
            avg_reward_difference = 0
        else:
            max_reward_difference = numpy.nanmax(self.level_novelty_indicators[REWARD_PROB])
            avg_reward_difference = numpy.nanmean(self.level_novelty_indicators[REWARD_PROB])

        dataframe = pandas.DataFrame(columns=['ColumnName.HAS_NOVEL_OBJECT',
                                              'ColumnName.MAX_REWARD_DIFFERENCE',
                                              'ColumnName.AVG_REWARD_DIFFERENCE',
                                              'ColumnName.MAX_PDDL_INCONSISTENCY',
                                              'ColumnName.AVG_PDDL_INCONSISTENCY'])

        X = dataframe.append({
            'ColumnName.HAS_NOVEL_OBJECT': has_unknown_object,
            'ColumnName.MAX_REWARD_DIFFERENCE': max_reward_difference,
            'ColumnName.AVG_REWARD_DIFFERENCE': avg_reward_difference,
            'ColumnName.MAX_PDDL_INCONSISTENCY': max_pddl_inconsistency,
            'ColumnName.AVG_PDDL_INCONSISTENCY': avg_pddl_inconsistency
        }, ignore_index=True)

        detection_threshold = settings.SB_LEVEL_NOVELTY_DETECTION_ENSEMBLE_THRESHOLD

        if detection_threshold is None:
            is_novel_df = rf.predict(X)
        else:
            predicted_probabilities = rf.predict_proba(X)
            print(predicted_probabilities)
            is_novel_df = (predicted_probabilities[:, 1] >= detection_threshold).astype('int')
            print(is_novel_df)

        if is_novel_df[0] == 0:
            return False
        else:
            return True

    def _infer_novelty_existence(self):

        print("Novelty existence is {}".format(self.novelty_existence))
        if (self.novelty_existence == 0) or (self.novelty_existence == 1):
            self._new_novelty_likelihood = self.novelty_existence
            return

        '''looks at the history of detections in previous levels and returns true when novelty has been detected for 3 contiguous episodes'''
        # self.novelty_detections.append(self._detect_level_novelty())
        self.novelty_detections.append(self._detect_level_novelty_with_ensemble())
        if (not self._new_novelty_likelihood) and len(self.novelty_detections) > 2:
            self._new_novelty_likelihood = self.novelty_detections[-1] and self.novelty_detections[-2] and \
                                           self.novelty_detections[-3]

    def _handle_end_of_level(self, success):
        """ This is called when a level has ended, either in a win or a loss outcome """
        self._need_to_repair = self.made_plan and not success
        self.completed_levels.append(success)
        # self._infer_novelty_existence()  TODO: this is the novelty detection, turn back on when done with video
        self.stats_for_level[NOVELTY_LIKELIHOOD] = bool(self._new_novelty_likelihood)
        self.stats_for_level[PDDL_PROB] = self.level_novelty_indicators[PDDL_PROB]
        self.stats_for_level[REWARD_PROB] = self.level_novelty_indicators[REWARD_PROB]
        self.stats_for_level[UNKNOWN_OBJ] = self.level_novelty_indicators[UNKNOWN_OBJ]
        self.stats_for_level['novelty_detections'] = self.novelty_detections
        logger.info("[hydra_agent_server] :: Level novelty indicators {}".format(self.level_novelty_indicators))
        logger.info("[hydra_agent_server] :: Novelty detections from new code {}".format(self.novelty_detections))
        logger.info(
            "[hydra_agent_server] :: Novelty likelihood from the new code {}".format(self._new_novelty_likelihood))
        logger.info("[hydra_agent_sever] :: Novelty existence notification is {}".format(self.novelty_existence))
        logger.info("[hydra_agent_server] :: Level {} Complete - WIN={}".format(self.current_level, success))
        logger.info("[hydra_agent_server] :: Cumulative planning time only = {}".format(str(self.cumulative_plan_time)))
        logger.info("[hydra_agent_server] :: Planning effort percentage = {}\n".format(
            str((self.cumulative_plan_time / (time.perf_counter() - self.overall_plan_time)))))
        logger.info("[hydra_agent_server] :: Overall time to attempt level {} = {}\n\n".format(self.current_level, str(
            (time.perf_counter() - self.overall_plan_time))))
        self.cumulative_plan_time = 0
        self.overall_plan_time = time.perf_counter()

        self.agent_stats.append(self.stats_for_level)

        self.current_level = self.env.sb_client.load_next_available_level()
        self.perception.new_level = True
        self.shot_num = 0

        self.stats_for_level = dict()
        self.level_novelty_indicators = {
            PDDL_PROB: list(),
            UNKNOWN_OBJ: list(),
            REWARD_PROB: list(),
            PLAN: list()
        }
        # time.sleep(1)
        self.novelty_existence = self.env.sb_client.get_novelty_info()
        print("Given novelty existence is {}".format(self.novelty_existence))
        time.sleep(2 / settings.SB_SIM_SPEED)

    def handle_game_playing(self, observation, raw_state):
        """ Handle what happens when the agent receives a PLAYING request"""

        processed_state = self.perception.process_state(raw_state)
        observation.state = processed_state
        self.choose_action(observation)

    def choose_action(self, observation: ScienceBirdsObservation):
        """ Choose which action to perform in the current obseration """
        processed_state = observation.state
        self.shot_num += 1
        if processed_state:
            logger.info("[hydra_agent_server] :: Invoking Planner".format())
            simplifications = settings.SB_PLANNER_SIMPLIFICATION_SEQUENCE.copy()
            simplifications.reverse()
            plan = []
            try:
                while len(simplifications) > 0 and (len(plan) == 0 or plan[0].action_name == "out of memory"):
                    simplification = simplifications.pop()
                    start_time = time.perf_counter()
                    plan, self.made_plan = self.planner.make_plan(processed_state, simplification)
                    plan_time = (time.perf_counter() - start_time)
                    self.stats_for_level[f'simplification level time {simplification}'] = plan_time
                    self.cumulative_plan_time += plan_time
                    logger.info(
                        "[hydra_agent_server] :: Problem simplification {} planning time: {}".format(simplification,
                                                                                                     str(plan_time)))
            except Exception as e:
                logger.error("Planner threw an exception. Exception details:\n {}".format(e))

            if len(plan) == 0 or plan[0].action_name == "out of memory":  # TODO FIX THIS
                plan = []
                plan.append(self.__get_default_action(processed_state))
                self.made_plan = False
            self.level_novelty_indicators[PLAN].append(plan)
            timed_action = plan[0]

            ## TAP UPDATE
            if len(plan) > 1 and "bird_action" in plan[1].action_name:
                t_time = int((plan[1].start_at - plan[0].start_at) * 1000)
                sb_action = self.meta_model.create_sb_action(timed_action, processed_state, tap_timing=t_time)
                logger.info("[hydra_agent_server] :: Taking tap action: [{}] {}ms after launch. ".format(
                    str(plan[1].action_name), str(t_time)))
            else:
                sb_action = self.meta_model.create_sb_action(timed_action, processed_state)

            observation.action = sb_action
            # plot_expected_trace_for_obs(self.meta_model, observation, settings.SB_DELTA_T)

            raw_state, reward = self.env.act(sb_action)
            observation.reward = reward
            if self.stats_for_level.get('rewards_per_shot'):
                self.stats_for_level['rewards_per_shot'].append(reward)
            else:
                self.stats_for_level['rewards_per_shot'] = [reward]

            observation.intermediate_states = list(self.env.intermediate_states)
            self.perception.process_observation(observation)
            if settings.DEBUG:
                observation.log_observation(
                    '{}_{}_{}_{}'.format(self.current_level, self.shot_num, self.trial_timestamp,
                                         self.planner.current_problem_prefix))
            logger.info("[hydra_agent_server] :: Reward {} Game State {}".format(reward, raw_state.game_state))
            # time.sleep(5)
        else:
            logger.info("Perception Failure performing default shot")
            plan = PddlPlusPlan()
            plan.append(self.__get_default_action(processed_state))
            sb_action = self.meta_model.create_sb_action(plan[0], processed_state)
            pddl_state = self.meta_model.create_pddl_problem(processed_state).get_init_state()
            self.latest_angle = self.meta_model.action_time_to_angle(plan[0].start_at, pddl_state)
            raw_state, reward = self.env.act(sb_action)
            logger.info("[hydra_agent_server] :: Reward {} Game State {}".format(reward, raw_state.game_state))

    def __get_default_action(self, state: ProcessedSBState):
        """ A default action taken by the Hydra agent if planning fails"""

        logger.info("[hydra_agent_server] :: __get_default_action")
        self.stats_for_level['Default action used'] = True
        problem = self.meta_model.create_pddl_problem(state)
        pddl_state = PddlPlusState(problem.init)
        unknown_objs = state.novel_objects()
        if unknown_objs:
            logger.info("unknown objects in {},{} : {}".format(self.current_level,
                                                               self.planner.current_problem_prefix,
                                                               unknown_objs.__str__()))
        try:
            active_bird = pddl_state.get_active_bird()
        except:
            active_bird = None  # TODO catch only appropriate exception
        try:
            pig_x, pig_y = get_random_pig_xy(problem)  # TODO: try to shoot at _all_ pigs
            if settings.SB_DEFAULT_SHOT == 'RANDOM_PIG':
                min_angle, max_angle = estimate_launch_angle(self.planner.meta_model.get_slingshot(state),
                                                             Point2D(pig_x, pig_y), self.meta_model)
                default_time = self.meta_model.angle_to_action_time(min_angle, pddl_state)
            elif settings.SB_DEFAULT_SHOT == 'RANDOM':
                default_angle = random.randint(pddl_state.numeric_fluents[('angle',)],
                                               pddl_state.numeric_fluents[('max_angle',)])
                default_time = self.meta_model.angle_to_action_time(default_angle, pddl_state)
            elif settings.SB_DEFAULT_SHOT == 'PLANNING':
                min_angle, max_angle = estimate_launch_angle(self.planner.meta_model.get_slingshot(state),
                                                             Point2D(pig_x, pig_y), self.meta_model)
                default_time = self.meta_model.angle_to_action_time(min_angle, pddl_state)
            else:
                logger.info("invalid setting for SB_DEFAULT_SHOT, taking default angle of 20")
                default_time = self.meta_model.angle_to_action_time(20, pddl_state)
        except:
            if unknown_objs:
                logger.info(
                    "Unable to shoot at a random pig, shooting at unknown object")  # TODO carch only appropriate exception
                target_x, target_y = get_x_coordinate(unknown_objs[0]), \
                                     get_y_coordinate(unknown_objs[0],
                                                      self.meta_model.get_ground_offset(
                                                          self.meta_model.get_slingshot(state)))
                min_angle, max_angle = estimate_launch_angle(self.planner.meta_model.get_slingshot(state),
                                                             Point2D(target_x, target_y), self.meta_model)
                default_time = self.meta_model.angle_to_action_time(min_angle, pddl_state)
            else:
                logger.info("Unable to shoot at a random pig, no unknown objects, shooting at 20 degrees")
                default_time = self.meta_model.angle_to_action_time(20, pddl_state)
        return TimedAction("pa-twang %s" % active_bird, default_time)

    def set_env(self, env):
        """Probably bad to have two pointers here"""
        self.env = env

    def run_next_action(self):
        """ Runs the agent until it performs an action"""

        while True:
            evaluation_done = self.main_loop(max_actions=1)
            if self.observations[-1].action is not None:
                return
            if evaluation_done:
                return

    def find_last_obs(self):
        """ Finds the last observations of the game. That is, the last observation that has intermediate states.
        TODO: Is this the best way to implement this?"""

        i = -1
        if len(self.observations) == 0:
            return None
        while self.observations[i].intermediate_states is None:
            i = i - 1
            if len(self.observations) + i < 0:
                return None
        return self.observations[i]


class RepairingSBHydraAgent(SBHydraAgent):
    """ Repairing Hydra agent for the SB domain """

    def __init__(self, env=None, agent_stats=None):
        super().__init__(env, agent_stats=agent_stats)
        # Repair and detection variables
        if agent_stats is None:
            agent_stats = list()
        settings.NOVELTY_POSSIBLE = True
        self.revision_attempts = 0
        self.meta_model_repair = ScienceBirdsMetaModelRepair(self.meta_model)

    def reinit(self):
        super().reinit()
        self.revision_attempts = 0

    def process_final_observation(self):
        """ This is called after winning or losing a level. """
        # self.stats_for_level[NOVELTY_LIKELIHOOD]=self._new_novelty_likelihood
        # The consistency score per level for this level is the mean over the consistency scored of this level's observations
        # self.pddl_prob_per_level.insert(0,
        # sum(self.stats_for_level[PDDL_PROB]) / len(self.stats_for_level[PDDL_PROB]))
        pass

    def handle_evaluation_terminated(self):
        """ Handle what happens when the agent receives a EVALUATION_TERMINATED request"""
        # store info and disconnect the agent as the evaluation is finished
        self.process_final_observation()
        return super().handle_evaluation_terminated()

    def handle_game_won(self):
        self.process_final_observation()
        super().handle_game_won()

    def handle_game_lost(self):
        self.process_final_observation()
        super().handle_game_lost()

    def handle_game_playing(self, observation, raw_state):
        """ Handle what happens when the agent receives a PLAYING request"""
        last_obs = self.find_last_obs()
        if last_obs is not None:
            if REPAIR_CALLS not in self.stats_for_level:
                self.stats_for_level[REPAIR_CALLS] = 0
            if REPAIR_TIME not in self.stats_for_level:
                self.stats_for_level[REPAIR_TIME] = 0

            # Check if we should repair
            logger.info("checking for repair...")
            should_repair = self._need_to_repair  # self.should_repair(last_obs)
            print("Should repair is {}".format(should_repair))
            if should_repair and (settings.NO_REPAIR == False):
                self.repair_meta_model(last_obs)

        logger.info("Novelty likelihood is {}".format(self._new_novelty_likelihood))

        super().handle_game_playing(observation, raw_state)

    def repair_meta_model(self, last_obs):
        """ Repair the metamodel based on the last observation"""
        self.stats_for_level[REPAIR_CALLS] = self.stats_for_level[REPAIR_CALLS] + 1
        self.revision_attempts += 1
        logger.info("Initiating repair number {}".format(self.revision_attempts))
        start_repair_time = time.time()
        try:
            repair, consistency = self.meta_model_repair.repair(self.meta_model, last_obs, delta_t=settings.SB_DELTA_T)
            repair_description = ["Repair %s, %.2f" % (fluent, repair[i])
                                  for i, fluent in enumerate(self.meta_model_repair.fluents_to_repair)]
            logger.info(
                "Repair done! Consistency: %.2f, Repair:\n %s" % (consistency, "\n".join(repair_description)))
        except:
            # TODO: fix this hack, catch correct exception
            import traceback
            traceback.print_exc()
            logger.info("Repair failed!")
        self.stats_for_level[REPAIR_TIME] = time.time() - start_repair_time

    def should_repair(self, observation: ScienceBirdsObservation):
        """
        Checks if the current model should be repaired
        If we are going to repair for a level, it will be a repair with the first shot's observations for that level.
        """

        # If novelty existence is given, use the given
        if self.revision_attempts >= settings.HYDRA_MODEL_REVISION_ATTEMPTS:
            return False

        if self.novelty_existence != NOVELTY_EXISTENCE_NOT_GIVEN:
            return self.novelty_existence == 1

        if self._new_novelty_likelihood:
            return True
