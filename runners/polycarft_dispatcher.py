
from worlds.polycraft_world import *
from agent.hydra_agent import HydraAgent
class PolycraftDispatcher():
    def __init__(self, agent: HydraAgent):
        self.env = None
        self.agent = agent

    def experiment_start(self):
        # Load polycarft world
        config = dict()
        config["headless"]=False
        config["levels"] = "levels"
        self.env = Polycraft(launch=True, server_config=config)

    def trial_start(self, trial_number: int, novelty_description: dict):
        # Run multiple levels
        print("?")
        novelty_description = None
        levels = [] # TODO Populate me
        for level_num, level in enumerate(levels):
            self.env.init_selected_level(level)
            self.episode_start(level_num, novelty_description)
            self.episode_end()

    def episode_start(self, trial_number: int, novelty_description: dict):
        # Run the agent in a single level until done
        current_state = self.env.get_current_state()
        while True:

            # Check if level is done
            if current_state.is_terminal():
                return

            # Agent chooses an action
            action = self.agent.choose_action(current_state)

            # Perform the action
            next_state, step_cost = self.env.act(action)

            current_state = next_state

            # If detected novelty, report it as follows
            # self.env.poly_client.REPORT_NOVELTY(level, confidence, user_msg)


    def episode_end(self):
        # Cleanup
        print("?")

    def trial_end(self):
        # Cleanup
        print("?")

    def experiment_end(self):
        # Cleanup
        self.env.kill()