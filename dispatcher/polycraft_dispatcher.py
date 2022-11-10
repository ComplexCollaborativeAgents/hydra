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
    world:Polycraft
    agent:PolycraftHydraAgent
    trials:Dict[str, List]
    results:Dict[str, Dict[str, List]]

    def __init__(self, agent: PolycraftHydraAgent):
        super().__init__(agent)
        self.world = None
        self.agent = agent

        # Current loaded trials, with key/value pairs: <trial_id>/<List of paths to .json levels>
        self.trials = {}

        # Results of the most recent trial set
        self.results = {}

        #
        self.trial_params = {
            'num_trials'
        }

    def report_novelty(self, level:str, confidence:str, user_msg:str) -> dict:
        """ Wrapper function to pass to the agent for reporting novelty

        Returns:
            dict: REPORT_NOVELTY message acknowledgement
        """

        # TODO: move "world" operations out of hydra agent and use this as callback for when needed
        return self.world.poly_client.REPORT_NOVELTY(level=level, confidence=confidence, user_msg=user_msg)

    def run(self, trial_sets:Dict[str, List[str]]):
        """Run an evaluation using the provided trial sets

        Args:
            trial_sets (Dict[str, List[str]]): Sets of trials
        """
        self.set_trial_sets(trial_sets)
        self.run_experiment(self.trials)
        self.run_trials()

    def set_trial_sets(self, trial_sets:Dict[str, List[str]]):
        """Sets the set of trials to run

        Args:
            trial_sets (Dict[str, List[str]]): Sets of trials
        """
        self.trials = trial_sets

    def run_experiment(self, trials:List[str]=[], standalone:bool=False):
        """Perform required setup and start the experiment. 

        Args:
            trials (List[str], optional): a list that contains string paths towards each trial directory. Defaults to [].
            standalone (bool, optional): boolean flag that signals to run without launching Polycraft.  If true, requires a running Polycraft instance before startup. Defaults to False.
        """

        # Load polycraft world
        mode = ServerMode.CLIENT
        if not standalone:
            mode = ServerMode.SERVER
        self.world = Polycraft(polycraft_mode=mode)
        self.trials = {}

        if len(trials) > 0:
            # Setup trials
            self.trials = self._get_trial_paths(trials)

    def _get_trial_paths(self, trials:List[str]) -> Dict[str, List[str]]:
        """Collect a set of trials from the list of paths.  Used for initializing provided trial sets.

        Args:
            trials (List[str]): List of paths to directories that contain .json levels

        Returns:
            Dict[str, List[str]]: A dictionary with key/value pairings: <trial_id>/<List of .json level paths>
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
        """Run the trials setup in "setup_trials"
        """

        trial_num = 0
        print(self.trials)
        for trial_id, trial_levels in self.trials.items():
            # TODO: include novelty characterization? (also detection stats too)
            self.run_trial(trial_id, trial_num, {}, trial_levels)
            self.cleanup_trial(trial_id, trial_num)
            trial_num += 1

    def run_trial(self, trial_id: str, trial_number:int, novelty_description: dict, levels: list):
        """ Run a trial that consists of a set of levels

        Args:
            trial_id (str): The identifier for this trial
            trial_number (int): The number of the trial in the trial set
            novelty_description (dict): description of the novelty
            levels (list): List of levels to run
        """

        logger.info("------------ [{}] TRIAL {} START ------------".format(trial_id, trial_number))

        novelty_description = None

        self.results[trial_id] = []

        # Iterate through each level
        for level_num, level in enumerate(levels):
            self.world.init_selected_level(level)
            
            result = self.run_episode(level_num, trial_id, trial_number, novelty_description)

            self.results[trial_id].append(result)
            
            self.cleanup_episode()

    def _setup_for_new_level(self):
        self.world.init_state_information()
        logger.info("Waiting for level to load...")
        time.sleep(10.0) # Wait for level to load
        logger.info("Performing exploratory actions...")
        self.agent.episode_init(self.world) # Agent performing exploratory actions
        return self.world.get_current_state()

    def run_episode(self, level_number: int, trial_id:str, trial_number: int, novelty_description: dict) -> dict:
        """Run the agent in a single level until done

        Args:
            level_number (int): The number of the level in the trial
            trial_id (str): The id of the trial
            trial_number (int): The number of the trial in the set
            novelty_description (dict): description of the novelty

        Returns:
            dict: Description of the novelty
        """

        logger.info("------------ [{}] EPISODE {} START ------------".format(trial_number, level_number))
        start_time = time.time()

        current_state = self._setup_for_new_level()
        
        while True:
            reached_time = EPISODE_TIME_LIMIT < time.time() - start_time

            if reached_time:
                logger.info("------------ Ran out of time! ------------")

            # Check if level is done
            if current_state.is_terminal() or reached_time:
                detection = self.agent.episode_end()
                
                stats = self.agent.get_agent_stats()[-1]

                result = {
                    'trial_id': trial_id,
                    'trial_number': trial_number,
                    'level': self.world.current_level,
                    'episode_number': level_number,
                    'step_cost': self.world.get_level_total_step_cost(),
                }

                result.update(vars(detection))
                result.update(vars(stats))

                return result

            # Agent chooses an action and performs it
            action = self.agent.choose_action(current_state, self.world)
            # TODO: split self.agent.do() into upkeep (agent) and performing actions (world)
            next_state, step_cost = self.agent.do(action, self.world)

            current_state = next_state

            # If detected novelty, report it as follows
            # self.world.poly_client.REPORT_NOVELTY(level, confidence, user_msg)

            # Check if the process has timeout

    def cleanup_episode(self) -> dict:
        """Cleanup agent/world processes after episode completion

        Returns:
            dict: _description_
        """
        return

    def cleanup_trial(self, trial_id:str, trial_number:int):
        """ Perform cleanup for the trial

        Args:
            trial_id (str): The id of the trial
            trial_number (int): The number of the trial in the set
        """
        logger.info("------------ [{}] {} TRIAL END ------------".format(trial_number, trial_id))

    def cleanup_experiment(self):
        """Clean up the experiment and any lingering agent/environment processes
        """

        logger.info("------------ EXPERIMENT END ------------")
        logger.info(json.dumps(self.results, indent=4))

        if self.world is not None:
            self.world.kill()
            self.world = None
