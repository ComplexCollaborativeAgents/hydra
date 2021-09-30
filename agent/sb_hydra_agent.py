import datetime

import settings
from agent.hydra_agent import logger, NN_PROB, PDDL_PROB, NOVELTY_EXISTANCE_NOT_GIVEN, NOVELTY_LIKELIHOOD
from agent.planning.sb_planner import SBPlanner
from agent.repair.meta_model_repair import *

# TODO: Maybe push this to the settings file? then every module just adds a logger
from agent.repair.sb_repair import ScienceBirdsConsistencyEstimator, ScienceBirdsMetaModelRepair
from agent.gym_hydra_agent import REPAIR_CALLS, REPAIR_TIME, logger
from state_prediction.anomaly_detector_fc_multichannel import FocusedSBAnomalyDetector
from utils.point2D import Point2D
from worlds.science_birds_interface.client.agent_client import GameState
from agent.hydra_agent import HydraAgent
logging.basicConfig(format='%(name)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("hydra_agent")

# Flags from ANU
NOVELTY_EXISTANCE_NOT_GIVEN = -1 # The self.novelty_existance value indicating that novelty detection is not given by the environment

# stats_per_level dictionary keys
NN_PROB = "nn_novelty_likelihood"
PDDL_PROB = "pddl_novelty_likelihood"
NOVELTY_LIKELIHOOD = "novelty_likelihood"
UNKNOWN_OBJ = "unknown_object"
UNDEFINED = None


class SBHydraAgent(HydraAgent):
    '''
    Probably needs to subclass for each domain. We will cross that bridge when we get there
    '''
    def __init__(self,env=None, agent_stats = list()):
        logger.info("[hydra_agent_server] :: Agent Created")

        super().__init__(planner = SBPlanner(ScienceBirdsMetaModel()),
                         meta_model_repair=None)

        # Default values
        self.env = env # agent always has a pointer to its environment
        if env is not None:
            env.sb_client.set_game_simulation_speed(settings.SB_SIM_SPEED)
        #self.perception = Perception()
        #self.completed_levels = []
        #self.observations = []
        #self.cumulative_plan_time = 0.0
        #self.overall_plan_time = 0.0
        #self.novelty_likelihood = 0.0
        #self.novelty_existence = -1
        #self.novel_objects = []
        self.agent_stats = agent_stats
        #self.shot_num = 0
        #self.trial_timestamp = datetime.datetime.now().strftime("%y%m%d%H%M%S")
        #self.stats_for_level = dict()
        #self.nn_prob_per_level = []
        #self.pddl_prob_per_level = []
        self.consistency_estimator = ScienceBirdsConsistencyEstimator()
        self.detector = FocusedSBAnomalyDetector()
        self.current_level = 0
        self.novelty_detections = list()

        ## SM: added a method to cleanup code
        self.initialize_processing_state_variables()


    def initialize_processing_state_variables(self):
        self.perception = Perception()
        self.completed_levels=[]
        self.observations=[]

        self.novelty_likelihood = 0.0
        self.novelty_existence = -1
        self.novel_objects = []

        self.cumulative_plan_time=0.0
        self.overall_plan_time=0.0
        self.shot_num=0
        self.trial_timestamp = datetime.datetime.now().strftime("%y%m%d%H%M%S")

        self.stats_for_level = dict()
        self.level_novelty_indicators = {
            NN_PROB: list(),
            PDDL_PROB: list(),
            UNKNOWN_OBJ: list()
        }

        self.nn_prob_per_level = []
        self.pddl_prob_per_level = []


    def reinit(self):
        logging.info('Reinit...')
        self.env.history = []
        #self.perception = Perception()
        self.meta_model = ScienceBirdsMetaModel()
        self.planner = SBPlanner(self.meta_model) # TODO: Discuss this w. Wiktor & Matt
        self.initialize_processing_state_variables()

        #self.completed_levels = []
        #self.observations = []
        #self.novelty_likelihood = 0.0
        #self.novel_objects = []
        #self.cumulative_plan_time = 0.0
        #self.overall_plan_time = 0.0
        #self.novelty_existence = -1
        # self.agent_stats = list() # TODO: Discuss this
        #self.shot_num = 0
        #self.trial_timestamp = datetime.datetime.now().strftime("%y%m%d%H%M%S")
        #self.stats_for_level = dict()
        #self.nn_prob_per_level = []
        #self.pddl_prob_per_level = []

    def main_loop(self,max_actions=1000):
        ''' Runs the agent. Returns False if the evaluation has not ended, and True if it has ended.'''

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
                self.handle_game_playing(observation, raw_state)
                if (settings.NOVELTY_POSSIBLE):
                    self._compute_novelty_likelihood(observation)
                    self._record_novelty_indicators(observation)
                    print(self.level_novelty_indicators)
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
            elif raw_state.game_state.value == GameState.NEWTRIAL.value:
                self.handle_new_trial()
            elif raw_state.game_state.value == GameState.MAIN_MENU.value:
                self.handle_main_menu()
            else:
                logger.info("[hydra_agent_server] :: Unexpected state.game_state.value {}".format(raw_state.game_state.value))
                assert False
            t+=1

            self.observations.append(observation)
        return False

    def handle_main_menu(self):
        logger.info("unexpected main menu page, reload the level : %s" % self.current_level)
        self.current_level = self.env.sb_client.load_next_available_level()
#        self.novelty_existence = self.env.sb_client.get_novelty_info()

    def handle_new_trial(self):
        ''' Handle what happens when the agent receives a NEWTRIAL request'''

        # DO something to start a fresh agent for a new training set
        (time_limit, interaction_limit, n_levels, attempts_per_level, mode, seq_or_set,
         allowNoveltyInfo) = self.env.sb_client.ready_for_new_set()
        logger.info("New Trial Request Received. Refresh agent.")
        self.reinit()
        self.current_level = 0
        self.training_level_backup = 0
        change_from_training = True
        self.current_level = self.env.sb_client.load_next_available_level()
        #self.novelty_existence = self.env.sb_client.get_novelty_info()

    def handle_request_novelty_likelihood(self):
        ''' Handle what happens when the agent receives a REQUESTNOVELTYLIKELIHOOD request'''

        logger.info("[hydra_agent_server] :: Requesting Novelty Likelihood. Novelyy likelihood is {}".format(
            self.novelty_likelihood))
        novelty_likelihood = self.novelty_likelihood
        non_novelty_likelihood = 1 - novelty_likelihood
        #placeholders for novelty information
        if len(self.novel_objects)>0:
            ids = set([int(object_id_str) for object_id_str in self.novel_objects])
            novelty_description = "Unknown type of objects detected"
        else:
            ids = {1,2,3}
            novelty_description = "Uncharacterized novelty"
        novelty_level = 0

        self.env.sb_client.report_novelty_likelihood(novelty_likelihood, non_novelty_likelihood,ids,novelty_level,novelty_description)

    def handle_evaluation_terminated(self):
        ''' Handle what happens when the agent receives a EVALUATION_TERMINATED request'''

        # store info and disconnect the agent as the evaluation is finished
        self._handle_end_of_level(True)
        logger.info("Evaluation complete.")
        return True

    def handle_new_training_set(self):
        ''' Handle what happens when the agent receives a NEWTRAININGSET request'''

        # DO something to start a fresh agent for a new training set
        (time_limit, interaction_limit, n_levels, attempts_per_level, mode, seq_or_set,
         allowNoveltyInfo) = self.env.sb_client.ready_for_new_set()
        self.current_level = 0
        self.training_level_backup = 0
        change_from_training = True
        self.current_level = self.env.sb_client.load_next_available_level()
        #self.novelty_existence = self.env.sb_client.get_novelty_info()

    def handle_game_lost(self):
        ''' Handle what happens when the agent receives a LOST request'''
        self._handle_end_of_level(False)

    def handle_game_won(self):
        ''' Handle what happens when the agent receives a WON request'''
        self._handle_end_of_level(True)
        return self.cumulative_plan_time, self.overall_plan_time

    def compute_novelty_detection_info(self, observation):
        try:
            cnn_novelty, cnn_prob = self.detector.detect(observation)
        except:
            logging.info('CNN Index out of Bounds in game playing')
            cnn_prob = 1.0  # TODO: Think about this design choice

        if settings.NO_PDDL_CONSISTENCY:
            pddl_prob = 1.0
        else:
            pddl_prob = check_obs_consistency(observation, self.meta_model, self.consistency_estimator)

        return cnn_prob, pddl_prob

    def _record_novelty_indicators(self, observation: ScienceBirdsObservation):
        logging.info("Computing novelty likelihood...")

        if self.novelty_existence in [0,1]:
            self.level_novelty_indicators[NN_PROB].append(UNDEFINED)
            self.level_novelty_indicators[PDDL_PROB].append(UNDEFINED)
            self.level_novelty_indicators[UNKNOWN_OBJ].append(UNDEFINED)
            return

        if observation.hasUnknownObj():
            self.level_novelty_indicators[NN_PROB].append(UNDEFINED)
            self.level_novelty_indicators[PDDL_PROB].append(UNDEFINED)
            self.level_novelty_indicators[UNKNOWN_OBJ].append(True)
            self.novel_objects = observation.get_novel_object_ids()
            return

        self.level_novelty_indicators[UNKNOWN_OBJ].append(False)
        try:
            cnn_novelty, cnn_prob = self.detector.detect(observation)
        except:
            logging.info('CNN Index out of Bounds in game playing')
            cnn_prob = 1.0  # TODO: Think about this design choice

        self.level_novelty_indicators[NN_PROB].append(cnn_prob)

        if settings.NO_PDDL_CONSISTENCY:
            pddl_prob = UNDEFINED
        else:
            pddl_prob = check_obs_consistency(observation, self.meta_model, self.consistency_estimator)
        self.level_novelty_indicators[PDDL_PROB].append(pddl_prob)


    def _compute_novelty_likelihood(self, observation: ScienceBirdsObservation):
        ''' Computes the novelty likelihood for the given observation
        Also updates the stats_for_level object with the computed novelty probability by the two models.  '''

        logging.info('Computing novelty likelihood...')
        #
        if NN_PROB not in self.stats_for_level:
            self.stats_for_level[NN_PROB]=[]
        if PDDL_PROB not in self.stats_for_level:
            self.stats_for_level[PDDL_PROB]=[]

        # if novelty existences is given by the experiment framework - no need to run the fancy models
        if self.novelty_existence in  [0,1]:
            self.stats_for_level[NN_PROB].append(self.novelty_existence)
            self.stats_for_level[PDDL_PROB].append(self.novelty_existence)
            self.novelty_likelihood = self.novelty_existence
        else:
            assert self.novelty_existence==NOVELTY_EXISTANCE_NOT_GIVEN # The flag denoting that we do not get novelty info from the environment

            if observation.hasUnknownObj():
                self.stats_for_level[NN_PROB].append(1.0)
                self.stats_for_level[PDDL_PROB].append(1.0)
                self.novelty_likelihood = 1.0
                self.novel_objects = observation.get_novel_object_ids()
            else:
                try:
                    cnn_novelty, cnn_prob = self.detector.detect(observation)
                except:
                    logging.info('CNN Index out of Bounds in game playing')
                    cnn_prob=1.0 # TODO: Think about this design choice

                self.stats_for_level[NN_PROB].append(cnn_prob)

                if settings.NO_PDDL_CONSISTENCY:
                    pddl_prob = 1.0
                else:
                    pddl_prob = check_obs_consistency(observation, self.meta_model, self.consistency_estimator)
                self.stats_for_level[PDDL_PROB].append(pddl_prob)

                # If we already played at least two levels and novelty keeps being detected, mark this as a very high novelty likelihood
                cnn_prediction = cnn_prob > self.detector.threshold
                pddl_prediction = pddl_prob > settings.SB_CONSISTENCY_THRESHOLD
                enough_levels_completed = len(self.completed_levels)>1
                if enough_levels_completed:
                    level_lost = not self.completed_levels[-1]
                    last_level_prediction = self.nn_prob_per_level[0] > self.detector.threshold or \
                                            self.pddl_prob_per_level[0] > self.meta_model_repair.consistency_threshold

                    last_last_level_prediction = self.nn_prob_per_level[1] > self.detector.threshold or \
                                            self.pddl_prob_per_level[1] > self.meta_model_repair.consistency_threshold
                else:
                    level_lost = False
                    last_level_prediction = False
                    last_last_level_prediction = False

                detected = cnn_prediction \
                           and pddl_prediction \
                           and enough_levels_completed \
                           and level_lost \
                           and last_level_prediction \
                           and last_last_level_prediction
                if detected:
                    self.novelty_likelihood = 1.0
                # else:
                # TODO: Think about how to set a non-binary value for novelty_likelihood
                #     Ideal: compute novelty prob from (cnn_prob, pddl_prob)

                logger.info("ShotNum={}, Level={}, Novelty detected={}, NN prediction={}, PDDL prediction={}, enough_levels={}, last_level_lost={}, last_pred={}, last_last_pred={}".
                    format(self.shot_num, len(self.completed_levels), detected, cnn_prediction, pddl_prediction, enough_levels_completed, level_lost, last_level_prediction, last_last_level_prediction))

        # Record current novelty likelihood estimate
        self.stats_for_level[NOVELTY_LIKELIHOOD]=self.novelty_likelihood

    def _detect_level_novelty(self):
        is_novel = False
        has_new_object = False

        if True in self.level_novelty_indicators[UNKNOWN_OBJ]:
            has_new_object = True

        mean_nn_probability = sum(self.level_novelty_indicators[NN_PROB])/len(self.level_novelty_indicators[NN_PROB])
        mean_pddl_inconsistency = sum(self.level_novelty_indicators[PDDL_PROB])/len(self.level_novelty_indicators[PDDL_PROB])
        are_level_observations_divergent = (mean_nn_probability > self.detector.threshold) and ( mean_pddl_inconsistency > self.meta_model_repair.consistency_threshold)

        is_novel = has_new_object or are_level_observations_divergent
        return is_novel



    def _infer_novelty_existence(self):
        '''looks at the history of detections in previous levels and returns true when novelty has been detected for 3 contiguous episodes'''
        self.novelty_detections.append(self._detect_level_novelty())
        self.novelty_likelihood = self.novelty_detections[-1] and self.novelty_detections[-2] and self.novelty_detections [-3]

    def _handle_end_of_level(self, success):
        ''' This is called when a level has ended, either in a win or a lose our come '''
        self.completed_levels.append(success)
        self._infer_novelty_existence()
        print(self.novelty_detections)
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
            NN_PROB: list(),
            PDDL_PROB: list(),
            UNKNOWN_OBJ: list()
        }
        # time.sleep(1)
        self.novelty_existence = self.env.sb_client.get_novelty_info()
        time.sleep(2 / settings.SB_SIM_SPEED)


    def handle_game_playing(self, observation, raw_state):
        ''' Handle what happens when the agent receives a PLAYING request'''

        processed_state = self.perception.process_state(raw_state)
        observation.state = processed_state
        self.choose_action(observation)

    def choose_action(self, observation : ScienceBirdsObservation):
        ''' Choose which action to perform in the current obseration '''
        processed_state = observation.state
        self.shot_num += 1
        if processed_state:
            logger.info("[hydra_agent_server] :: Invoking Planner".format())
            simplifications = settings.SB_PLANNER_SIMPLIFICATION_SEQUENCE.copy()
            simplifications.reverse()
            plan = []
            while len(simplifications) > 0 and (len(plan) == 0 or plan[0].action_name == "out of memory"):
                simplification = simplifications.pop()
                start_time = time.perf_counter()
                plan = self.planner.make_plan(processed_state, simplification)
                plan_time = (time.perf_counter() - start_time)
                self.cumulative_plan_time += plan_time
                logger.info("[hydra_agent_server] :: Problem simplification {} planning time: {}".format(simplification,
                                                                                                         str(plan_time)))
            if len(plan) == 0 or plan[0].action_name == "out of memory":  # TODO FIX THIS
                plan = []
                plan.append(self.__get_default_action(processed_state))
            timed_action = plan[0]
            logger.info("[hydra_agent_server] :: Taking action: {}".format(str(timed_action.action_name)))
            sb_action = self.meta_model.create_sb_action(timed_action, processed_state)
            raw_state, reward = self.env.act(sb_action)
            observation.reward = reward
            observation.action = sb_action
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
            raw_state, reward = self.env.act(sb_action)
            logger.info("[hydra_agent_server] :: Reward {} Game State {}".format(reward, raw_state.game_state))

    def __get_default_action(self, state : ProcessedSBState):
        ''' A default action taken by the Hydra agent if planning fails'''

        logger.info("[hydra_agent_server] :: __get_default_action")
        problem = self.meta_model.create_pddl_problem(state)
        pddl_state = PddlPlusState(problem.init)
        unknown_objs = state.novel_objects()
        if unknown_objs:
            logger.info("unknown objects in {},{} : {}".format(self.current_level,
                        self.planner.current_problem_prefix,unknown_objs.__str__()))
        try:
            active_bird = pddl_state.get_active_bird()
        except:
            active_bird = None
        try:
            pig_x, pig_y = get_random_pig_xy(problem)
            if settings.SB_DEFAULT_SHOT == 'RANDOM_PIG':
                min_angle, max_angle = estimate_launch_angle(self.planner.meta_model.get_slingshot(state), Point2D(pig_x, pig_y), self.meta_model)
                default_time = self.meta_model.angle_to_action_time(min_angle, pddl_state)
            elif settings.SB_DEFAULT_SHOT == 'RANDOM':
                default_angle = random.randint(pddl_state.numeric_fluents[('angle',)], pddl_state.numeric_fluents[('max_angle',)])
                default_time = self.meta_model.angle_to_action_time(default_angle, pddl_state)
            elif settings.SB_DEFAULT_SHOT == 'PLANNING':
                min_angle, max_angle = estimate_launch_angle(self.planner.meta_model.get_slingshot(state),
                                                             Point2D(pig_x, pig_y), self.meta_model)
                default_time = self.meta_model.angle_to_action_time(min_angle, pddl_state)
            else:
                logger.info("invalid setting for SB_DEFAULT_SHOT, taking default angle of 20")
                default_time = self.meta_model.angle_to_action_time(20, pddl_state)
        except:
            logger.info("Unable to shoot at a random pig, taking default angle of 20")
            default_time = self.meta_model.angle_to_action_time(20, pddl_state)
        return TimedAction("pa-twang %s" % active_bird, default_time)

    def set_env(self,env):
        '''Probably bad to have two pointers here'''
        self.env = env

    def run_next_action(self):
        ''' Runs the agent until it performs an action'''

        while True:
            evaluation_done = self.main_loop(max_actions=1)
            if self.observations[-1].action is not None:
                return
            if evaluation_done == True:
                return

    def find_last_obs(self):
        ''' Finds the last observations of the game. That is, the last observation that has intermediate states.
        TODO: Is this the best way to implement this?'''

        i = -1
        if len(self.observations)==0:
            return None
        while self.observations[i].intermediate_states is None:
            i=i-1
            if len(self.observations)+i<0:
                return None
        return self.observations[i]


class RepairingSBHydraAgent(SBHydraAgent):
    ''' Repairing Hydra agent for the SB domain '''

    def __init__(self,env=None, agent_stats = list()):
        super().__init__(env, agent_stats=agent_stats)
        # Repair and detection variables
        self.revision_attempts = 0
        self.meta_model_repair = ScienceBirdsMetaModelRepair(self.meta_model)

    def reinit(self):
        super().reinit()
        self.revision_attempts = 0

    def process_final_observation(self):
        ''' This is called after winning or losing a level. '''
        self.stats_for_level[NOVELTY_LIKELIHOOD]=self.novelty_likelihood
        # The consistency score per level for this level is the mean over the consistency scored of this level's observations

        print(self.stats_for_level)
        self.nn_prob_per_level.insert(0,
                                      sum(self.stats_for_level[NN_PROB]) / len(self.stats_for_level[NN_PROB]))
        self.pddl_prob_per_level.insert(0,
                                      sum(self.stats_for_level[PDDL_PROB]) / len(self.stats_for_level[PDDL_PROB]))



    def handle_evaluation_terminated(self):
        ''' Handle what happens when the agent receives a EVALUATION_TERMINATED request'''
        # store info and disconnect the agent as the evaluation is finished
        self.process_final_observation()
        return super().handle_evaluation_terminated()

    def handle_game_won(self):
        self.process_final_observation()
        super().handle_game_won()

    def handle_game_lost(self):
        self.process_final_observation()
        super().handle_game_lost()

    ''' Handle what happens when the agent receives a PLAYING request'''
    def handle_game_playing(self, observation, raw_state):
        last_obs = self.find_last_obs()
        if last_obs!=None:
            if REPAIR_CALLS not in self.stats_for_level:
                self.stats_for_level[REPAIR_CALLS] = 0
            if REPAIR_TIME not in self.stats_for_level:
                self.stats_for_level[REPAIR_TIME] = 0

            # Check if we should repair
            logger.info("checking for repair...")
            if self.should_repair(last_obs) and settings.NO_REPAIR==False:

                self.repair_meta_model(last_obs)

        # In should_repair, self.novelty_likelihood is turned into a 1 if past 3 consistency scores are high enough (and is otherwise < 1).
        # 1 should be considered True, anything else should be considered False
        logger.info("Novelty likelihood is {}".format(self.novelty_likelihood))

        super().handle_game_playing(observation, raw_state)

    def repair_meta_model(self, last_obs):
        ''' Repair the meta model based on the last observation'''
        self.stats_for_level[REPAIR_CALLS] = self.stats_for_level[REPAIR_CALLS] + 1
        self.revision_attempts += 1
        logger.info("Initiating repair number {}".format(self.revision_attempts))
        start_repair_time = time.time()
        try:
            repair, consistency = self.meta_model_repair.repair(self.meta_model, last_obs)
            repair_description = ["Repair %s, %.2f" % (fluent, repair[i])
                                  for i, fluent in enumerate(self.meta_model_repair.fluents_to_repair)]
            logger.info(
                "Repair done! Consistency: %.2f, Repair:\n %s" % (consistency, "\n".join(repair_description)))
        except:
            # TODO: fix this hack
            logger.info("Repair failed!")
        self.stats_for_level[REPAIR_TIME] = time.time() - start_repair_time

    def should_repair(self, observation: ScienceBirdsObservation):
        '''
        Checks if the current model should be repaired
        If we are going to repair for a level, it will be a repair with the first shot's observations for that level.
        '''

        # If novelty existance is given, use the given
        if self.revision_attempts >= settings.HYDRA_MODEL_REVISION_ATTEMPTS:
            return False

        if self.novelty_existence != NOVELTY_EXISTANCE_NOT_GIVEN:
            return self.novelty_existence==1

        if observation.hasUnknownObj():
            return True

        if NN_PROB not in self.stats_for_level:
            return False #TODO: Design choice: wait for the second shot to repair

        cnn_prob = self.stats_for_level[NN_PROB][-1]
        pddl_prob = self.stats_for_level[PDDL_PROB][-1]

        # Try to repair only after 2 levels have passed & only after the first shot of the level # TODO: Rethink this design choice
        if len(self.completed_levels) < 2 or len(self.stats_for_level[NN_PROB]) != 1:
            return False

        logger.info("CNN novelty likelihoods last shot: %.3f, previous problem: %.3f, two problems ago: %.3f, last problem solved? %s" % (cnn_prob,
                                                                                                                                          self.nn_prob_per_level[0],
                                                                                                                                          self.nn_prob_per_level[1],
                                                                                                                                          self.completed_levels[-1]))

        if self.novelty_likelihood==1.0 and \
                pddl_prob > self.meta_model_repair.consistency_threshold and \
                cnn_prob > self.detector.threshold:
            return True

        return False