import datetime
import logging
import pathlib
import time
import os
from typing import Dict, List
from xml.etree import ElementTree as ET

import settings
from agent.sb_hydra_agent import SBHydraAgent
from dispatcher.hydra_dispatcher import Dispatcher
from worlds.science_birds import SBState, ScienceBirds
from worlds.science_birds_interface.client.agent_client import GameState

EPISODE_TIME_LIMIT = 17 * 60    # 17 minutes

# Logger
logging.basicConfig(format='%(name)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("polycraft_dispatcher")
logger.setLevel(logging.INFO)


class SBDispatcher(Dispatcher):
    """ SuperClass for hydra agent dispatcher. Interfaces connecting hydra
            agnet and environment for running an evaluation.

    Attributes:
    """
    world: ScienceBirds                # trial/evaluation that is being run
    agent: SBHydraAgent           # hydra agent being tested
    trial_timestamp: str
    results:Dict[str, List[dict]]
    launch: bool            # Whether or not to launch the SB server
    host: str               # What host server to connect to
    current_level: int      # level number assigned by ScienceBirds
    episode_num: int        # Current episode number
    trial_num: int          # Current trial number
    export_trial: bool      # Whether or not to export trials
    current_trial_id: str   # Current trial id
    config_file: str        # Current config file in use

    def __init__(self, agent:SBHydraAgent, launch:bool=True, host:str=None):
        self.trial_timestamp = datetime.datetime.now().strftime("%y%m%d%H%M%S")
        self.agent = agent
        self.trials = {}
        self.results = {}
        self.launch = launch
        self.host = host
        self.world = None   # Set to none so that the environment can be loaded with the correct levels on trial start
        self.current_level = 0
        self.episode_num = 0
        self.trial_num = 0
        self.current_trial_id = ""
        self.config_file = ""

    def report_novelty(self, state: SBState) -> dict:
        """ Return a dictionary with novelty detail and stats

        Returns:
            Dict[str, Any]: Dictionary with novelty details

        """
        
        novelty_likelihood, non_novelty_likelihood, ids, novelty_level, novelty_description = self.agent.report_novelty(state, self.world)

        self.world.sb_client.report_novelty_likelihood(novelty_likelihood, non_novelty_likelihood, ids, novelty_level, novelty_description)

        return {}


    def set_trial_info(self, trial_id:str, config:str):
        self.current_trial_id = trial_id
        self.config_file = config
        logger.info(f"Next trial ({trial_id}) will be run with {config}")

    def run(self, trials_to_run:int = -1):
        """ Run an evaluation for self.agent in self.world
        """
        self.episode_num = 0

        self.run_experiment()
        self.cleanup_experiment()


    def run_experiment(self):
        """ Run an experiment
        """
        logger.info("[hydra_agent_server] :: Entering main loop")
        logger.info("[hydra_agent_server] :: Delta t = {}".format(str(settings.SB_DELTA_T)))
        logger.info("[hydra_agent_server] :: Simulation speed = {}\n\n".format(str(settings.SB_SIM_SPEED)))
        logger.info("[hydra_agent_server] :: Planner memory limit = {}".format(str(settings.SB_PLANNER_MEMORY_LIMIT)))
        logger.info("[hydra_agent_server] :: Planner timeout = {}s\n\n".format(str(settings.SB_TIMEOUT)))

        self.run_trial()
        self.cleanup_trial()
        self.trial_num += 1

    def cleanup_experiment(self):
        """ Perform cleanup after running the experiment
        """
        if self.world is not None and self.launch:
            self.world.kill()
            self.world = None

    def run_trial(self):
        """ Run a trial for the experiment
        """

        if self.world is None and self.launch:
            # science birds world object handles config = None
            self.world = ScienceBirds(launch=self.launch, config=self.config_file, host=self.host)

        self.trial_timestamp = datetime.datetime.now().strftime("%y%m%d%H%M%S")
        self.agent.trial_start()
        trial_progress = 0

        self.results[self.current_trial_id] = []

        while True:
            state = self.world.get_current_state()

            if state.game_state == GameState.MAIN_MENU:
                self.current_level = self.world.sb_client.load_next_available_level()
            elif state.game_state == GameState.NEWTRIAL:    # We should not be reaching this state?
                self.world.sb_client.ready_for_new_set()
                self.agent.trial_start()
                self.current_level = self.world.sb_client.load_next_available_level()
            elif state.game_state == GameState.NEWTRAININGSET:
                self.world.sb_client.ready_for_new_set()
                self.current_level = self.world.sb_client.load_next_available_level()
            elif state.game_state == GameState.EVALUATION_TERMINATED:
                self.cleanup_trial()
                break
            else:
                self.agent.episode_init(self.current_level, self.world)

                result = self.run_episode(state)
                self.results[self.current_trial_id].append(result)
                self.cleanup_episode()
                trial_progress += 1
                self.episode_num += 1

    def cleanup_trial(self):
        """ Perform clean after running an experiment.
        """
        self.agent.trial_end()
        if self.world is not None and self.launch:
            self.world.kill()
            self.world = None
        time.sleep(10)

    def run_episode(self, starting_state:SBState) -> dict:
        """ Run one episode of the trial
        """
        
        logger.info("------------ [{}] EPISODE {} START ------------".format(self.trial_num, self.episode_num))
        # start_time = time.time()
        state = starting_state

        while True:
            # reached_time = EPISODE_TIME_LIMIT < time.time() - start_time

            if state.game_state == GameState.REQUESTNOVELTYLIKELIHOOD:
                novelty_likelihood, non_novelty_likelihood, ids, novelty_level, novelty_description = self.agent.report_novelty()
                self.world.sb_client.report_novelty_likelihood(novelty_likelihood, non_novelty_likelihood, ids, novelty_level,
                                                               novelty_description)

                time.sleep(5 / settings.SB_SIM_SPEED)
                state = self.world.get_current_state()
            
            if state.is_terminal():
                success = state.game_state == GameState.WON or state.game_state == GameState.EVALUATION_TERMINATED
                detection = self.agent.episode_end(success)

                stats = self.agent.get_agent_stats()[-1]

                result = {
                    'trial_id': self.current_trial_id,
                    'trial_number': self.trial_num,
                    'level': self.current_level,    # TODO: extract level name info from config file?
                    'episode_number': self.episode_num,
                }

                result.update(vars(detection))
                result.update(vars(stats))

                self.current_level = self.world.sb_client.load_next_available_level()

                return result

            if state.game_state == GameState.PLAYING:
                self.world.sb_client.batch_ground_truth(1, 1)
                
                action = self.agent.choose_action(state)

                state, step_cost = self.agent.do(action, self.world, self.trial_timestamp)

    def cleanup_episode(self):
        """ Perform cleanup after running an episode of a trial
        """


