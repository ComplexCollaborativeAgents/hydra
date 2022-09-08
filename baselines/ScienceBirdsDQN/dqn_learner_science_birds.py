import logging, pathlib
import math

import settings
import worlds.science_birds as sb
from worlds.science_birds_interface.client.agent_client import GameState
from agent.perception.perception import Perception
import random

from runners.run_sb_stats import prepare_config

SB_DATA_PATH = pathlib.Path(settings.ROOT_PATH) / 'data' / 'science_birds'
SB_CONFIG_PATH = SB_DATA_PATH / 'config'
SB_BIN_PATH = pathlib.Path(settings.SCIENCE_BIRDS_BIN_DIR) / 'linux'


class QNet():
    pass


class DQNSBAgent():
    def __init__(self, environment):
        self.log = logging.getLogger(__name__).getChild('DQNSBAgent')
        self.env = environment
        self.perception = Perception()
        pass


    def main_loop(self, max_actions=10000):
        while True:
            raw_state = self.env.get_current_state()

            if raw_state.game_state.value == GameState.NEWTRIAL.value:
                self.handle_new_trial()

            if raw_state.game_state.value == GameState.NEWTRAININGSET.value:
                self.handle_new_training_set()

            if raw_state.game_state.value == GameState.PLAYING.value:
                raw_state = self.env.get_current_state()
                action = self.choose_action(raw_state)
                raw_state, reward = self.env.act(action)

            if raw_state.game_state.value == GameState.REQUESTNOVELTYLIKELIHOOD.value:
                self.handle_novelty_request()

    def handle_new_trial(self):
        ## reset agent
        self.env.sb_client.ready_for_new_set()
        self.env.sb_client.load_next_available_level()

    def handle_new_training_set(self):
        self.env.sb_client.ready_for_new_set()
        self.env.sb_client.load_next_available_level()

    def choose_action(self, raw_state):
        processed_state = self.perception.process_state(raw_state)
        print(processed_state.objects)
        angle = random.randrange(0, 90, 1)
        action = self.generate_sb_action(processed_state, angle)
        return action

    def generate_sb_action(self, processed_state, angle):
        tp = sb.ScienceBirds.trajectory_planner
        ref_point = tp.get_reference_point(processed_state.sling)
        release_point = tp.find_release_point(processed_state.sling, math.radians(angle))
        sb_action = sb.SBShoot(release_point.X, release_point.Y, 3000, ref_point.X, ref_point.Y)
        return sb_action


    def handle_game_lost(self):
        pass

    def handle_game_won(self):
        pass

    def handle_novelty_request(self):
        pass


def get_levels_list_for_novelty(novelty, novelty_type):
    samples = 10
    pattern = '9001_Data/StreamingAssets/Levels/novelty_level_{}/type{}/Levels/*.xml'.format(novelty, novelty_type)
    levels_path = SB_BIN_PATH
    levels = list(levels_path.glob(pattern))
    number_samples = len(levels)
    if samples is not None:
        number_samples = min(number_samples, samples)
    levels = levels[20:20 + number_samples]

    template = SB_CONFIG_PATH / 'test_config.xml'
    prepare_config(template, config, levels, None)
    print(config)

if __name__ == '__main__':
    config = SB_CONFIG_PATH/'stats_config.xml'
    novelty = 0
    novelty_type = 2
    pattern = '9001_Data/StreamingAssets/Levels/novelty_level_{}/type{}/Levels/*.xml'.format(novelty, novelty_type)
    print("running level: " + pattern)
    get_levels_list_for_novelty(novelty, novelty_type)
    env = sb.ScienceBirds(None, launch=True, config=config)
    agent = DQNSBAgent(environment=env)
    agent.main_loop()

