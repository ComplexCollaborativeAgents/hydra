
from worlds.polycraft_world import *
from agent.hydra_agent import HydraAgent


class PolycraftDispatcher():
    def __init__(self, agent: HydraAgent):
        self.env = None
        self.agent = agent

        self.results = []

    def experiment_start(self, config: dict):

        # Load polycarft world
        self.env = Polycraft(launch=True, server_config=config)

    def trial_start(self, trial_number: int, novelty_description: dict):
        ''' Run multiple levels '''

        novelty_description = None
        levels = [] # TODO Populate me

        trial_results = []

        for level_num, level in enumerate(levels):
            self.env.init_selected_level(level)
            
            trial_results.append(self.episode_start(level_num, novelty_description))
            
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

            # Agent chooses an action
            action = self.agent.choose_action(current_state)

            # Perform the action
            next_state, step_cost = self.env.act(action)    # Note this returns step cost for the action

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

        if self.env is not None:
            self.env.kill()
            self.env = None


if __name__ == '__main__':
    dispatcher = PolycraftDispatcher()
