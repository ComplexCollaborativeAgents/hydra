import pathlib
import logging
from typing import List, Dict

from worlds.polycraft_world import *
from agent.polycraft_hydra_agent import PolycraftHydraAgent
from dispatcher.hydra_dispatcher import Dispatcher

# Logger
logging.basicConfig(format='%(name)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("polycraft_dispatcher")
logger.setLevel(logging.INFO)

EPISODE_TIME_LIMIT = 17 * 60    # 17 minutes

class PolycraftDispatcher(Dispatcher):
    env:Polycraft
    agent:PolycraftHydraAgent
    trials:Dict[str, List]
    results:Dict[str, Dict[str, List]]

    def __init__(self, agent: PolycraftHydraAgent):
        super().__init__(agent)
        self.env = None
        self.agent = agent

        # Current loaded trials, with key/value pairs: <trial_id>/<List of paths to .json levels>
        self.trials = {}

        # Results of the most recent trial set
        self.results = {}

        #
        self.trial_params = {
            'num_trials'
        }

    def set_trial_sets(self, trial_sets:Dict[str, List[str]]):
        """
        Sets the set of trials to run
        """
        self.trials = trial_sets

    def run_experiment(self, trials:List[str]=[], standalone:bool=False):
        ''' 
        Start the experiment.  
        :param standalone: boolean flag that signals to run without launching Polycraft.  If true, requires a running Polycraft instance before startup
        :param trials: a list that contains string paths towards each trial directory
        '''

        # Load polycraft world
        mode = ServerMode.CLIENT
        if not standalone:
            mode = ServerMode.SERVER
        self.env = Polycraft(polycraft_mode=mode)
        self.trials = {}

        if len(trials) > 0:
            # Setup trials
            self.trials = self._get_trial_paths(trials)

    def _get_trial_paths(self, trials:List[str]) -> Dict[str, List]:
        """
        Collect a set of trials from the list of paths.  Used for initializing provided trial sets.
        :param trials: List of paths to directories that contain .json levels
        :returns: A dictionary with key/value pairings: <trial_id>/<List of .json level paths>
        """
        trial_set = {}

        for trial in trials:
            # collect all trial filenames
            trial_set[trial] = []
            for level in os.listdir(trial): # Make sure to only add the .json files
                suffix = pathlib.Path(level).suffix
                if suffix == ".json":
                    trial_set[trial].append(os.path.join(trial, level))

        return trial_set

    def run_trials(self):
        """ Run trials setup in "setup_trials" """

        trial_num = 0
        for trial_id, trial_levels in self.trials.items():
            # TODO: include novelty characterization? (also detection stats too)
            self.run_trial(trial_id, trial_num, {}, trial_levels)
            self.cleanup_trial(trial_id, trial_num)
            trial_num += 1

    def run_trial(self, trial_id: str, trial_number:int, novelty_description: dict, levels: list):
        ''' Run multiple levels '''

        logger.info("------------ [{}] TRIAL {} START ------------".format(trial_id, trial_number))

        novelty_description = None

        self.results[trial_id] = []

        for level_num, level in enumerate(levels):
            self.env.init_selected_level(level)
            
            result = self.run_episode(level_num, trial_id, trial_number, novelty_description)

            self.results[trial_id].append(result)
            
            self.cleanup_episode()

    def run_episode(self, level_number: int, trial_id:str, trial_number: int, novelty_description: dict):
        ''' Run the agent in a single level until done '''

        logger.info("------------ [{}] EPISODE {} START ------------".format(trial_number, level_number))
        start_time = time.time()

        current_state = self.env.get_current_state()
        self.agent.start_level(self.env) # Agent performing exploratory actions
        while True:
            novelty = self.agent.novelty_existence  # NOTE: This may be subject to change

            reached_time = EPISODE_TIME_LIMIT < time.time() - start_time

            if reached_time:
                logger.info("------------ Ran out of time! ------------")

            # Check if level is done
            if current_state.is_terminal() or reached_time:
                planning_times = str(self.agent.stats_for_level['planning times'])
                repair_calls = self.agent.stats_for_level['repair_calls']
                repair_time = str(self.agent.stats_for_level['repair_time'])
                novelty_likelihood = self.agent.stats_for_level['novelty_likelihood']

                return {
                    'trial_id': trial_id,
                    'trial_number': trial_number,
                    'level': self.env.current_level,
                    'episode_number': level_number,
                    'step_cost': self.env.get_level_total_step_cost(),
                    'novelty': novelty,
                    'novelty_description': novelty_description,
                    'novelty_likelihood': novelty_likelihood,
                    'passed': current_state.passed,
                    'planning_times': planning_times,
                    'repair_calls': repair_calls,
                    'repair_time': repair_time
                }

            # Agent chooses an action and performs it
            action = self.agent.choose_action(current_state)
            next_state, step_cost = self.agent.do(action, self.env)

            current_state = next_state

            # If detected novelty, report it as follows
            # self.env.poly_client.REPORT_NOVELTY(level, confidence, user_msg)

            # Check if the process has timeout

    def cleanup_episode(self) -> dict:
        ''' Cleanup level '''
        return

    def cleanup_trial(self, trial_id:str, trial_number:int):
        # Cleanup
        logger.info("------------ [{}] {} TRIAL END ------------".format(trial_number, trial_id))

    def cleanup_experiment(self):
        # Cleanup

        logger.info("------------ EXPERIMENT END ------------")
        logger.info(json.dumps(self.results, indent=4))

        if self.env is not None:
            self.env.kill()
            self.env = None
