import datetime
import logging
import pickle
import random
import time
from typing import List, Set, Tuple

import numpy as np
import pandas

import settings
from agent.consistency.nyx_pddl_simulator import NyxPddlPlusSimulator
from agent.consistency.fast_pddl_simulator import CachingPddlPlusSimulator
from agent.consistency.sb_episode_log import SBEpisodeLog
from agent.hydra_agent import (NOVELTY_EXISTENCE_NOT_GIVEN, NOVELTY_LIKELIHOOD,
                               PDDL_INCONSISTENCY, HydraAgent)
from agent.perception.perception import Perception, ProcessedSBState
from agent.planning.nyx.syntax import constants
from agent.planning.pddl_plus import PddlPlusPlan, PddlPlusState, TimedAction
from agent.planning.sb_meta_model import (ScienceBirdsMetaModel,
                                          estimate_launch_angle,
                                          get_random_pig_xy, get_x_coordinate,
                                          get_y_coordinate)
from agent.planning.sb_planner import SBPlanner
from agent.repair.sb_consistency_estimators.sb_domain_consistency_estimator import \
    ScienceBirdsConsistencyEstimator
from agent.repair.sb_repair import ScienceBirdsMetaModelRepair
from agent.reward_estimation.reward_estimator import RewardEstimator
from utils.point2D import Point2D
from utils.stats import SBAgentStats, SBDetectionStats
from worlds.science_birds import SBAction, SBState, ScienceBirds

# stats_per_level dictionary keys
# NN_PROB = "nn_novelty_likelihood" this originally was the state-based detector written by UPenn
REWARD_DISCREPENCY = "reward_estimator_likelihood"
PDDL_INCONSISTENCY = "pddl_novelty_likelihood"
NOVELTY_LIKELIHOOD = "novelty_likelihood"
UNKNOWN_OBJ_EXISTS = "unknown_object"
PLAN = "plan"
UNDEFINED = None

ENSEMBLE_MODEL = "{}/model/ensemble_may2022.pkl".format(settings.ROOT_PATH)

logger = logging.getLogger("Science Birds")


class SBHydraAgent(HydraAgent):
    meta_model: ScienceBirdsMetaModel
    planner: SBPlanner
    consistency_estimator: ScienceBirdsConsistencyEstimator
    meta_model_repair: ScienceBirdsMetaModelRepair

    current_state: SBState
    current_log: SBEpisodeLog       # NOTE: we have an episode log for every action taken
    current_stats: SBAgentStats
    current_novelty: SBDetectionStats

    agent_stats: List[SBAgentStats]
    novelty_stats: List[SBDetectionStats]
    
    current_level:int

    def __init__(self):
        super().__init__()


        self.current_stats = SBAgentStats(episode_start_time=time.perf_counter())
        self.current_novelty = SBDetectionStats()
        self.current_log = SBEpisodeLog()

        # Level tracking variables
        self.current_level = 0
        self.shot_num = 0

        # Simple repair tracker
        self._need_to_repair = False
        self.made_plan = False

        self._initialize_processing_state_variables()

    def _count_previous_detected_levels(self) -> int:
        """Count all levels novelty has been detected within a trial

        Returns:
            int: number of levels novelty detected
        """
        return sum(1 for detection in self.novelty_stats if detection.novelty_detected)

    def _detected_in_all_past_n(self, n:int) -> bool:
        """ Helper function that returns whether or not novelty was detected in all n of the past levels

        Args:
            n (int): Number of levels to search backwards to

        Returns:
            bool: Whether or not 
        """
        assert n != 0
        assert n < len(self.novelty_stats), f"Cannot find past {n} detections, as we only have stats for {len(self.novelty_stats)} levels"
        
        for i in range(-1, -1*n, -1):
            if not self.novelty_stats[i].novelty_detected:
                return False
        return True

    def _initialize_processing_state_variables(self):
        self.perception = Perception()
        self.completed_levels = []
        self.novel_objects = []
        self.level_novelty_indicators = {
            REWARD_DISCREPENCY: list(),
            PDDL_INCONSISTENCY: list(),
            UNKNOWN_OBJ_EXISTS: list(),
            PLAN: list()
        }

        self._new_novelty_likelihood = False

    def episode_init(self, level_num:int, world: ScienceBirds):
        """Perform setup for the agent at the beginning of an episode
        """
        self.perception.new_level = True
        self.shot_num = 0
        self.level_novelty_indicators = {
            PDDL_INCONSISTENCY: list(),
            UNKNOWN_OBJ_EXISTS: list(),
            REWARD_DISCREPENCY: list(),
            PLAN: list()
        }
        self.current_stats = SBAgentStats(episode_start_time=time.perf_counter())
        self.current_novelty = SBDetectionStats()
        self.current_log = SBEpisodeLog()
        
        self.current_level = level_num
        self.level_informed_novelty = world.sb_client.get_novelty_info()

        logger.info("Given novelty existence is {}".format(self.level_informed_novelty))
        time.sleep(2 / settings.SB_SIM_SPEED)

    def trial_start(self):
        """Perform setup for the agent at the start of a trial
        """
        logger.info("New Trial Request Received. Refresh agent.")
        self.meta_model = ScienceBirdsMetaModel()
        self.planner = SBPlanner(self.meta_model)
        self.reward_estimator = RewardEstimator()
        self.consistency_estimator = ScienceBirdsConsistencyEstimator()
        self.meta_model_repair = ScienceBirdsMetaModelRepair(self.meta_model, self.consistency_estimator)
        self.training_level_backup = 0


    def trial_end(self):
        """Perform setup for the agent at the end of a trial
        """
        # with open("novelty_stats_{}".format(datetime.datetime.now().strftime("%y%m%d%H%M%S")), 'w+') as f:
        #     for ns in self.novelty_stats:
        #         f.write(f"{str(ns)}\n")

        logger.info("Novelty detection stats are {}".format(self.novelty_stats))
        self.agent_stats = []
        self.novelty_stats = []
        self._initialize_processing_state_variables() 

    def episode_end(self, success: bool) -> SBDetectionStats:
        """Perform cleanup for the agent at the end of an episode

        Args:
            success (bool): Whether the level succeeded or not

        """

        # 
        self._need_to_repair = self.made_plan and not success
        self.completed_levels.append(success)

        self.current_stats.success = success
        self.current_novelty.pddl_prob = self.level_novelty_indicators[PDDL_INCONSISTENCY]
        self.current_novelty.reward_prob = self.level_novelty_indicators[REWARD_DISCREPENCY]
        self.current_novelty.unknown_obj = self.level_novelty_indicators[UNKNOWN_OBJ_EXISTS]
        self.current_novelty.novelty_detected = self._detect_if_current_episode_is_novel(success)

        logger.info("[hydra_agent_sever] :: Novelty existence notification is {}".format(self.level_informed_novelty))
        logger.info("[hydra_agent_server] :: Level {} Complete - WIN={}".format(self.current_level, success))
        logger.info("[hydra_agent_server] :: Cumulative planning time only = {}".format(str(self.current_stats.cumulative_plan_time)))
        logger.info("[hydra_agent_server] :: Planning effort percentage = {}\n".format(
            str((self.current_stats.cumulative_plan_time / (time.perf_counter() - self.current_stats.plan_total_time)))))
        self.perception.new_level = True
        self.novelty_stats.append(self.current_novelty)
        self.agent_stats.append(self.current_stats)

        return self.novelty_stats[-1]

    def __get_default_action(self, state: ProcessedSBState) -> TimedAction:
        """"""

        logger.info("[hydra_agent_server] :: __get_default_action")
        self.current_stats.default_action_used = True
        problem = self.meta_model.create_pddl_problem(state)
        pddl_state = PddlPlusState(problem.init)
        unknown_objs = state.novel_objects()
        if unknown_objs:
            logger.info("unknown objects in {},{} : {}".format(self.current_level,
                                                               self.planner.current_problem_prefix,
                                                               unknown_objs.__str__()))
        try:
            active_bird_id = int(pddl_state['active_bird'])
            active_bird = SBPlanner.get_bird(pddl_state, active_bird_id)
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
        logger.info(f"pa-twang {active_bird}, {default_time * float(pddl_state[('angle_rate',)])}")
        return TimedAction("pa-twang %s" % active_bird, default_time)

    def choose_action(self, state: SBState) -> SBAction:
        """Choose which action to perform in the given state.

        Args:
            state (SBState): State of the world

        Raises:
            NotImplementedError

        Returns:
            SBAction: action to take
        """
        
        processed_state = self.perception.process_state(state)
        self.current_log.state = processed_state
        self.current_stats.default_action_used = False

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
                    constants.SB_W_HELPFUL_ACTIONS = True
                    plan, self.made_plan = self.planner.make_plan(processed_state, simplification)
                    #### Additional statistics
                    if self.made_plan:
                        for act in self.planner.plan:
                            if not act[0] == constants.TIME_PASSING_ACTION:
                                self.current_stats.plan_action_length += 1
                        self.current_stats.plan_total_time = self.planner.plan[-1][1].time

                    ###
                    plan_time = (time.perf_counter() - start_time)
                    self.current_stats.planning_time = plan_time
                    self.current_stats.explored_states = self.planner.explored_states
                    self.current_stats.cumulative_plan_time += plan_time
                    logger.info(
                        "[hydra_agent_server] :: Problem simplification {} planning time: {}".format(simplification,
                                                                                                     str(plan_time)))
            except Exception as e:
                logger.error("Planner threw an exception. Exception details:\n {}".format(e))
                import traceback
                traceback.print_exc()

            if len(plan) == 0 or plan[0].action_name == "out of memory":  # TODO FIX THIS
                plan = []
                plan.append(self.__get_default_action(processed_state))
                self.made_plan = False
            self.level_novelty_indicators[PLAN].append(plan)
            timed_action = plan[0]

            ## TAP UPDATE
            if len(plan) > 1 and "bird_action" in plan[1].action_name:
                t_time = int((plan[1].start_at - plan[0].start_at) * 1000)
                sb_action = self.meta_model.create_sb_action(timed_action, processed_state, tap_timing=t_time - 500)
                # "-500" is a magic number - caused by some mismatch between the planner and the game.
                logger.info("[hydra_agent_server] :: Taking tap action: [{}] {}ms after launch. ".format(
                    str(plan[1].action_name), str(t_time)))
            else:
                sb_action = self.meta_model.create_sb_action(timed_action, processed_state)

            self.current_log.action = sb_action
            # plot_expected_trace_for_obs(self.meta_model, observation, settings.SB_DELTA_T)

            return sb_action
        else:
            logger.info("Perception Failure performing default shot")
            plan = PddlPlusPlan()
            plan.append(self.__get_default_action(processed_state))
            sb_action = self.meta_model.create_sb_action(plan[0], processed_state)
            pddl_state = self.meta_model.create_pddl_problem(processed_state).get_init_state()
            self.latest_angle = self.meta_model.action_time_to_angle(plan[0].start_at, pddl_state)
            return sb_action
    
    def _record_novelty_indicators(self):
        if self.level_informed_novelty == 0 or self.level_informed_novelty == 1:
            self.level_novelty_indicators[PDDL_INCONSISTENCY].append(UNDEFINED)
            self.level_novelty_indicators[UNKNOWN_OBJ_EXISTS].append(UNDEFINED)
            self.level_novelty_indicators[REWARD_DISCREPENCY].append(UNDEFINED)
            return

        self.level_novelty_indicators[UNKNOWN_OBJ_EXISTS].append(self.current_log.hasUnknownObj())
        if self.current_stats.default_action_used:
            self.level_novelty_indicators[PDDL_INCONSISTENCY].append(None)
        else:
            self.level_novelty_indicators[PDDL_INCONSISTENCY].append(self.consistency_estimator.consistency_from_simulator(self.current_log, self.meta_model,CachingPddlPlusSimulator(),self.meta_model.delta_t))
        #self.level_novelty_indicators[REWARD_DISCREPENCY].append(self.reward_estimator.compute_estimated_reward_difference(self.current_log))
        #self.level_novelty_indicators[PDDL_INCONSISTENCY].append(self.consistency.consistency_from_simulator(self.current_log, self.meta_model,NyxPddlPlusSimulator(),self.meta_model.delta_t))
        #self.level_novelty_indicators[REWARD_DISCREPENCY].append(self.reward_estimator.compute_estimated_reward_difference(self.current_log))


    def do(self, sb_action: SBAction, world: ScienceBirds, timestamp:str) -> Tuple[SBState, float]:
        """ Perform specified action within the environment, return the next state + resulting reward

        Args:
            sb_action (SBAction): Action to take
            world (ScienceBirds): Environment object
            timestamp (str): Trial timestamp

        Returns:
            Tuple[SBState, float]: Next state + reward
        """
        
        raw_state, reward = world.act(sb_action)
        self.current_log.reward = reward
        self.current_stats.rewards_per_shot.append(reward)

        self.current_log.intermediate_states = list(world.intermediate_states)
        
        if self.current_log.state is not None:
            self.perception.process_observation(self.current_log)
        if settings.DEBUG:
            self.current_log.log_observation(
                '{}_{}_{}_{}'.format(self.current_level, self.shot_num, timestamp,
                                        self.planner.current_problem_prefix))
        logger.info("[hydra_agent_server] :: Reward {} Game State {}".format(reward, raw_state.game_state))

        if settings.NOVELTY_POSSIBLE:
            self._record_novelty_indicators()
        self.episode_logs.append(self.current_log)
        return raw_state, reward

    def report_novelty(self) -> Tuple[float, float, Set[int], int, str]:
        """ Report detected novelty in detect novelty function.
        Wrapper for _detect_novelty function, can be called from the dispatcher
        """

        novelty_likelihood = 0.0
        non_novelty_likelihood = 1.0
        ids = set()
        novelty_level = 0
        novelty_description = "None"

        if len(self.novelty_stats) > 0:
            previous_episode_novelty_determination = self.novelty_stats[-1]
            if all(previous_episode_novelty_determination.unknown_obj):
                novelty_likelihood = 1.0
                non_novelty_likelihood = 0.0
                novelty_description = "novel object or agent"
        logger.info("[hydra_agent_server] :: Reporting novelty detection. Likelihood {}, characterization {}".format(novelty_likelihood, novelty_description))
        return novelty_likelihood, non_novelty_likelihood, ids, novelty_level, novelty_description

    def _detect_level_novelty_with_ensemble(self, success:bool) -> bool:
        """_summary_

        Args:
            success (bool): Whether or not the level succeeded

        Returns:
            bool: Novelty detection resulting from ensemble
        """
        
        with open(ENSEMBLE_MODEL, 'rb') as f:
            rf = pickle.load(f)

        if True in self.level_novelty_indicators[UNKNOWN_OBJ_EXISTS]:
            has_unknown_object = 1
        else:
            has_unknown_object = 0

        pddl_list = self.level_novelty_indicators[PDDL_INCONSISTENCY]

        if all(v is None for v in pddl_list):
            max_pddl_inconsistency = 1000
            avg_pddl_inconsistency = 1000
        else:
            max_pddl_inconsistency = np.nanmax(pddl_list)
            avg_pddl_inconsistency = np.nanmean(pddl_list)

        if len(self.level_novelty_indicators[REWARD_DISCREPENCY]) == 0:
            max_reward_difference = 0
            avg_reward_difference = 0
        else:
            max_reward_difference = np.nanmax(self.level_novelty_indicators[REWARD_DISCREPENCY])
            avg_reward_difference = np.nanmean(self.level_novelty_indicators[REWARD_DISCREPENCY])

        if success == True:
            status = 1
        else:
            status = 0

        dataframe = pandas.DataFrame(columns=['ColumnName.HAS_NOVEL_OBJECT',
                                              'ColumnName.MAX_REWARD_DIFFERENCE',
                                              'ColumnName.AVG_REWARD_DIFFERENCE',
                                              'ColumnName.MAX_PDDL_INCONSISTENCY',
                                              'ColumnName.AVG_PDDL_INCONSISTENCY',
                                              'ColumnName.PASS'])

        X = dataframe.append({
            'ColumnName.HAS_NOVEL_OBJECT': has_unknown_object,
            'ColumnName.MAX_REWARD_DIFFERENCE': max_reward_difference,
            'ColumnName.AVG_REWARD_DIFFERENCE': avg_reward_difference,
            'ColumnName.MAX_PDDL_INCONSISTENCY': max_pddl_inconsistency,
            'ColumnName.AVG_PDDL_INCONSISTENCY': avg_pddl_inconsistency,
            'ColumnName.PASS': status
        }, ignore_index=True)

        if settings.SB_LEVEL_NOVELTY_DETECTION_ENSEMBLE_THRESHOLD is None:
            is_novel_df = rf.predict(X)
            predicted_probabilities = rf.predict_proba(X)
        else:
            detection_threshold = settings.SB_LEVEL_NOVELTY_DETECTION_ENSEMBLE_THRESHOLD
            predicted_probabilities = rf.predict_proba(X)
            is_novel_df = (predicted_probabilities[:, 1] >= detection_threshold).astype('int')

        logger.info("[hydra_agent_server] :: Novelty detection input vector: {}; predicted probabilities: {}".format(
            X.to_dict(), predicted_probabilities))

        if is_novel_df[0] == 0:
            return False
        else:
            return True


    def _detect_if_current_episode_is_novel(self, success: bool) -> bool:
        """Given an episode log, determine using the methods available (consistency, perception, reward_prediction, etc) whether or not novelty is present

        Args:
            success (bool): Whether or not the agent succeeded this level

        Returns:
            bool: Whether or not novelty was detected
        """
        logger.info("Determining if a level is novel based on recorded novelty indicators")
        logger.info("Unknown object indicator is {}".format(self.level_novelty_indicators[UNKNOWN_OBJ_EXISTS]))
        logger.info("PDDL inconsistency indicator is {}".format(self.level_novelty_indicators[PDDL_INCONSISTENCY]))
        logger.info("Reward discrepency indicator is {}".format(self.level_novelty_indicators[REWARD_DISCREPENCY]))

        return False


class RepairingSBHydraAgent(SBHydraAgent):
    def __init__(self):
        super().__init__()
    
    def choose_action(self, state: SBState) -> SBAction:

        return super().choose_action(state)

    def episode_end(self, success: bool) -> SBDetectionStats:
        novelty_stats = super().episode_end(success)

        if self.should_repair():
            self.repair_meta_model()

        return novelty_stats
    
    def should_repair(self) -> bool:
        """ Choose if the agent should repair its meta model based on the given episode log.
            If we are going to repair for a level, it will be a repair with the first shot's observations for that level.

        Returns:
            bool: whether or not a repair should be initiated
        """
        # If novelty existence is given, use the given
        # logger.info(f"SHOULD_REPAIR: repair calls: {self.current_stats.repair_calls} >= {settings.HYDRA_MODEL_REVISION_ATTEMPTS}: {self.current_stats.repair_calls >= settings.HYDRA_MODEL_REVISION_ATTEMPTS}")
        if self.current_stats.repair_calls >= settings.HYDRA_MODEL_REVISION_ATTEMPTS:
            return False

        # If novelty existence is given
        if self.level_informed_novelty != NOVELTY_EXISTENCE_NOT_GIVEN:
            # logger.info(f"SHOULD_REPAIR: novelty given: {self.level_informed_novelty==1}")
            return self.level_informed_novelty == 1


        # logger.info(f"SHOULD_REPAIR: encountered: {self._new_novelty_likelihood}")
        # If we've encountered novelty before
        if self.current_novelty.pddl_prob[-1] and self.current_novelty.pddl_prob[-1] > 50:
            return True

        return False

    def repair_meta_model(self):
        """Call the repair object to repair the current meta model
        """
        self.current_stats.repair_calls += 1

        logger.info("Initiating repair number {}".format(self.current_stats.repair_calls))
        start_repair_time = time.time()

        try:
            repair, consistency = self.meta_model_repair.repair(self.current_log, delta_t=settings.SB_DELTA_T)
            repair_description = ["Repair %s, %.2f" % (fluent, repair[i])
                                  for i, fluent in enumerate(self.meta_model.repairable_constants)]
            self.current_stats.repair_description.append(repair_description)
            logger.info(
                "Repair done! Consistency: %.2f, Repair:\n %s" % (consistency, "\n".join(repair_description)))
            repaired_constant_description = ["Constant {}, repaired val {}".format(constant, self.meta_model.constant_numeric_fluents[constant])
                                             for i, constant in enumerate(self.meta_model.repairable_constants)]
            logger.info("[sb_hydra_agent] :: Updated meta model {} by {}".format(self.meta_model.uuid, repaired_constant_description))
        except:
            # TODO: fix this hack, catch correct exception
            import traceback
            traceback.print_exc()
            logger.info("Repair failed!")
        self.current_stats.repair_time = time.time() - start_repair_time