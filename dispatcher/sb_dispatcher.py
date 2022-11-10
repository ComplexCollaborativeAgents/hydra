import datetime
from typing import Dict
from dispatcher.hydra_dispatcher import Dispatcher
import settings
import time
import logging
from worlds.science_birds_interface.client.agent_client import GameState

from worlds.science_birds import SBState, ScienceBirds
from agent.sb_hydra_agent import SBHydraAgent

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

    def __init__(self, agent:SBHydraAgent):
        self.trial_timestamp = datetime.datetime.now().strftime("%y%m%d%H%M%S")
        self.agent = agent


    def report_novelty(self, state: SBState) -> dict:
        """ Return a dictionary with novelty detail and stats

        Returns:
            Dict[str, Any]: Dictionary with novelty details

        """
        
        novelty_likelihood, non_novelty_likelihood, ids, novelty_level, novelty_description = self.agent.report_novelty(state, self.world)

        self.world.sb_client.report_novelty_likelihood(novelty_likelihood, non_novelty_likelihood, ids, novelty_level, novelty_description)

        return {}


    def run(self):
        """ Run an evaluation for self.agent in self.world

        Raises:
            NotImplementedError
        """
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

    def cleanup_experiment(self):
        """ Perform cleanup after running the experiment
        """
        self.world.kill()

    def run_trial(self):
        """ Run a trial for the experiment
        """
        
        self.world.sb_client.ready_for_new_set()
        self.agent.trial_start()
        level_num = self.world.sb_client.load_next_available_level()
        trial_id = datetime.datetime.now().strftime("%y%m%d%H%M%S")

        while True:
            state = self.world.get_current_state()

            if state.game_state == GameState.MAIN_MENU.value:
                self.world.sb_client.load_next_available_level()
            elif state.game_state == GameState.EVALUATION_TERMINATED.value:
                self.cleanup_trial()
                break
            else:
                self.run_episode(level_num, trial_id, "0", state)
                self.cleanup_episode()

    def cleanup_trial(self):
        """ Perform clean after running an experiment.
        """
        self.agent.trial_end(self)

    def run_episode(self, level_number: int, trial_id:str, trial_number: int, starting_state:SBState) -> dict:
        """ Run one episode of the trial
        """
        
        logger.info("------------ [{}] EPISODE {} START ------------".format(trial_number, level_number))
        start_time = time.time()
        state = starting_state

        while True:
            reached_time = EPISODE_TIME_LIMIT < time.time() - start_time

            self.world.sb_client.batch_ground_truth(1, 1)

            if state.game_state.value == GameState.REQUESTNOVELTYLIKELIHOOD:
                self.agent.report_novelty()
            
            if state.is_terminal():
                detection = self.agent.episode_end()

                stats = self.agent.get_agent_stats()[-1]

                result = {
                    'trial_id': trial_id,
                    'trial_number': trial_number,
                    'level': self.world.current_level,
                    'episode_number': level_number,
                }

                result.update(vars(detection))
                result.update(vars(stats))

                return result

            action = self.agent.choose_action(state, self.world)
            next_state, step_cost = self.agent.do(action, self.world)

            state = next_state

    def cleanup_episode(self):
        """ Perform cleanup after running an episode of a trial
        """
        self.agent.episode_end()


