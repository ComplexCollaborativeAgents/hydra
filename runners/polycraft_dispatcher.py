import pathlib

from worlds.polycraft_world import *
from agent.hydra_agent import HydraAgent
from agent.polycraft_hydra_agent import PolycraftHydraAgent


class PolycraftDispatcher():
    def __init__(self, agent: PolycraftHydraAgent):
        self.env = None
        self.agent = agent

        self.trials = {}

        self.results = {}

    def experiment_start(self, trials: list = []):
        ''' Start the experiment.  server_config is a dict that contains configuration options for the Polycraft Server, and trials is a list that contains string paths towards each trial directory'''

        # Load polycarft world
        self.env = Polycraft(launch=True)
        self.trials = {}

        if len(trials) > 0:
            # Setup trials
            self.setup_trials(trials)

    def setup_trials(self, trials: list):

        # TODO: create a mix and match trial generator that has pre novelty and post novelty
        for trial in trials:
            # collect all trial filenames
            self.trials[trial] = []
            for level in os.listdir(trial): # Make sure to only add the .json files
                suffix = pathlib.Path(level).suffix
                if suffix == ".json":
                    self.trials[trial].append(os.path.join(trial, level))

    def run_trials(self):
        ''' Run trials setup in "setup_trials" '''

        trial_num = 0
        for trial_path, trial_levels in self.trials.items():
            self.trial_start(trial_num, {}, trial_levels)
            self.trial_end()

    def trial_start(self, trial_number: int, novelty_description: dict, levels: list):
        ''' Run multiple levels '''

        novelty_description = None

        self.results[trial_number] = []

        for level_num, level in enumerate(levels):
            self.env.init_selected_level(level)
            
            result = self.episode_start(level_num, trial_number, novelty_description)

            self.results[trial_number].append(result)
            
            self.episode_end()

    def episode_start(self, level_number: int, trial_number: int, novelty_description: dict):
        ''' Run the agent in a single level until done '''

        current_state = self.env.get_current_state()
        while True:
            novelty = 0

            # Check if level is done
            if current_state.is_terminal():
                return {
                    'trial_number': trial_number,
                    'level': self.env.current_level,
                    'episode_number': level_number,
                    'step_cost': self.env.get_level_total_step_cost(),
                    'novelty': novelty,
                    'novelty_description': novelty_description
                }

            # Agent chooses an action and performs it
            action = self.agent.choose_action(current_state)
            next_state, step_cost = self.agent.do(action, self.env)

            current_state = next_state

            # If detected novelty, report it as follows
            # self.env.poly_client.REPORT_NOVELTY(level, confidence, user_msg)

    def episode_end(self) -> dict:
        ''' Cleanup level '''

        return

    def trial_end(self):
        # Cleanup
        print("?")

    def experiment_end(self):
        # Cleanup

        print(self.results)

        self.trials = {}

        if self.env is not None:
            self.env.kill()
            self.env = None
