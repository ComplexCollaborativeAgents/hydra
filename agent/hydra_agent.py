from agent.planning.planner import Planner
import time

from agent.consistency.consistency_estimator import *
from worlds.science_birds_interface.client.agent_client import GameState
from agent.planning.pddl_meta_model import *
import datetime

# TODO: Maybe push this to the settings file? then every module just adds a logger
logging.basicConfig(format='%(name)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("hydra_agent")

class HydraAgent():
    '''
    Probably needs to subclass for each domain. We will cross that bridge when we get there
    '''
    def __init__(self,env=None, agent_stats = list()):
        logger.info("[hydra_agent_server] :: Agent Created")
        self.env = env # agent always has a pointer to its environment
        if env is not None:
            env.sb_client.set_game_simulation_speed(settings.SB_SIM_SPEED)
        self.perception = Perception()
        self.meta_model = MetaModel()
        self.planner = Planner(self.meta_model) # TODO: Discuss this w. Wiktor & Matt
        self.completed_levels = []
        self.observations = []
        self.novelty_likelihood = 0.0
        self.cumulative_plan_time = 0.0
        self.overall_plan_time = 0.0
        self.novelty_existence = -1
        self.agent_stats = agent_stats
        self.shot_num = 0
        self.trial_timestamp = datetime.datetime.now().strftime("%y%m%d%H%M%S")
        self.stats_for_level = dict()

    def reinit(self):
        self.env.history = []
        self.perception = Perception()
        self.meta_model = MetaModel()
        self.planner = Planner(self.meta_model) # TODO: Discuss this w. Wiktor & Matt
        self.completed_levels = []
        self.observations = []
        self.novelty_likelihood = 0.0
        self.cumulative_plan_time = 0.0
        self.overall_plan_time = 0.0
        self.novelty_existence = -1
        # self.agent_stats = list() # TODO: Discuss this
        self.shot_num = 0
        self.trial_timestamp = datetime.datetime.now().strftime("%y%m%d%H%M%S")
        self.stats_for_level = dict()

    ''' Runs the agent. Returns False if the evaluation has not ended, and True if it has ended.'''
    def main_loop(self,max_actions=1000):
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

    ''' Handle what happens when the agent receives a NEWTRIAL request'''
    def handle_new_trial(self):
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

    ''' Handle what happens when the agent receives a REQUESTNOVELTYLIKELIHOOD request'''
    def handle_request_novelty_likelihood(self):
        logger.info("[hydra_agent_server] :: Requesting Novelty Likelihood {}".format(
            self.novelty_likelihood))
        novelty_likelihood = self.novelty_likelihood
        non_novelty_likelihood = 1 - novelty_likelihood
        #placeholders for novelty information
        ids = {1,-2,-398879789}
        novelty_level = 0
        novelty_description = ""
        self.env.sb_client.report_novelty_likelihood(novelty_likelihood, non_novelty_likelihood,ids,novelty_level,novelty_description)

    ''' Handle what happens when the agent receives a EVALUATION_TERMINATED request'''
    def handle_evaluation_terminated(self):
        # store info and disconnect the agent as the evaluation is finished
        self.agent_stats.append(self.stats_for_level)
        logger.info("Evaluation complete.")
        return True

    ''' Handle what happens when the agent receives a NEWTRAININGSET request'''
    def handle_new_training_set(self):
        # DO something to start a fresh agent for a new training set
        (time_limit, interaction_limit, n_levels, attempts_per_level, mode, seq_or_set,
         allowNoveltyInfo) = self.env.sb_client.ready_for_new_set()
        self.current_level = 0
        self.training_level_backup = 0
        change_from_training = True
        self.current_level = self.env.sb_client.load_next_available_level()
        #self.novelty_existence = self.env.sb_client.get_novelty_info()

    ''' Handle what happens when the agent receives a LOST request'''
    def handle_game_lost(self):
        self._handle_end_of_level(False)

    ''' Handle what happens when the agent receives a WON request'''
    def handle_game_won(self):
        self._handle_end_of_level(True)
        return self.cumulative_plan_time, self.overall_plan_time

    def _handle_end_of_level(self, success):
        ''' This is called when a level has ended, either in a win or a lose our come '''
        self.completed_levels.append(success)
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
        # time.sleep(1)
        self.novelty_existence = self.env.sb_client.get_novelty_info()
        time.sleep(2 / settings.SB_SIM_SPEED)


    ''' Handle what happens when the agent receives a PLAYING request'''
    def handle_game_playing(self, observation, raw_state):
        processed_state = self.perception.process_state(raw_state)
        observation.state = processed_state
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
            if  len(plan) == 0 or plan[0].action_name == "out of memory":  # TODO FIX THIS
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
                observation.log_observation('{}_{}_{}_{}'.format(self.current_level,self.shot_num,self.trial_timestamp,self.planner.current_problem_prefix))
            logger.info("[hydra_agent_server] :: Reward {} Game State {}".format(reward, raw_state.game_state))
            # time.sleep(5)
        else:
            logger.info("Perception Failure performing default shot")
            plan = PddlPlusPlan()
            plan.append(self.__get_default_action(processed_state))
            sb_action = self.meta_model.create_sb_action(plan[0], processed_state)
            raw_state, reward = self.env.act(sb_action)
            logger.info("[hydra_agent_server] :: Reward {} Game State {}".format(reward, raw_state.game_state))

    ''' A default action taken by the Hydra agent if planning fails'''
    def __get_default_action(self, state : ProcessedSBState):
        logger.info("[hydra_agent_server] :: __get_default_action")
        problem = self.meta_model.create_pddl_problem(state)
        pddl_state = PddlPlusState(problem.init)
        unknown_objs = state.novel_objects()
        if unknown_objs:
            logger.info("unknown objects in {},{} : {}".format(self.current_level),
                        self.planner.current_problem_prefix,unknown_objs.__str__())
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

    ''' Runs the agent until it performs an action'''
    def run_next_action(self):
        while True:
            evaluation_done = self.main_loop(max_actions=1)
            if self.observations[-1].action is not None:
                return
            if evaluation_done == True:
                return

    ''' Finds the last observations of the game. That is, the last observation that has intermediate states. 
    TODO: Is this the best way to implement this?'''
    def find_last_obs(self):
        i = -1
        if len(self.observations)==0:
            return None
        while self.observations[i].intermediate_states is None:
            i=i-1
            if len(self.observations)+i<0:
                return None
        return self.observations[i]