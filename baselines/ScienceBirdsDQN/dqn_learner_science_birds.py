import logging, pathlib
import math, cv2

import settings
import worlds.science_birds as sb
from worlds.science_birds_interface.client.agent_client import GameState
from agent.perception.perception import Perception
from agent.reward_estimation.nn_utils.obs_to_imgs import SBObs_to_Imgs
import random

from runners.run_sb_stats import prepare_config

SB_DATA_PATH = pathlib.Path(settings.ROOT_PATH) / 'data' / 'science_birds'
SB_CONFIG_PATH = SB_DATA_PATH / 'config'
SB_BIN_PATH = pathlib.Path(settings.SCIENCE_BIRDS_BIN_DIR) / 'linux'


from matplotlib import pyplot as plt




class QNet():
    pass


class DQNSBAgent():
    def __init__(self, environment):
        self.log = logging.getLogger(__name__).getChild('DQNSBAgent')
        self.env = environment
        self.perception = Perception()
        self.state_coverter = SBObs_to_Imgs()

        self.done = False
        self.observation = None
        self.observation_ = None
        self.action = None
        self.reward = 0
        pass


    def main_loop(self, max_actions=10000):
        while True:
            raw_state = self.env.get_current_state()
            if raw_state.game_state.value == GameState.NEWTRIAL.value:
                self.handle_new_trial()

            elif raw_state.game_state.value == GameState.NEWTRAININGSET.value:
                self.handle_new_training_set(raw_state)

            elif raw_state.game_state.value == GameState.PLAYING.value:
                raw_state = self.env.get_current_state()
                self.observation = self.translate_to_observation(raw_state)
                self.action = self.choose_action(self.observation)
                sb_action = self.translate_to_sb_action(self.observation, self.action)
                raw_state, self.reward = self.env.act(sb_action)
                self.observation_ = self.translate_to_observation(raw_state)

            elif raw_state.game_state.value == GameState.REQUESTNOVELTYLIKELIHOOD.value:
                self.handle_novelty_request()

            elif raw_state.game_state.value == GameState.LOST.value:
                self.handle_end_of_level()

            elif raw_state.game_state.value == GameState.WON.value:
                self.handle_end_of_level()

            elif raw_state.game_state.value == GameState.EVALUATION_TERMINATED.value:
                self.handle_evaluation_terminated()
                break

            self.store_transition(self.observation, self.action, self.reward, self.observation_, self.done)
            self.learn()
            #self.observation = self.observation_


    def handle_new_trial(self):
        ## reset agent
        self.env.sb_client.ready_for_new_set()
        self.env.sb_client.load_next_available_level()
        self.raw_state = self.env.get_current_state()

    def handle_new_training_set(self, raw_state):
        self.done = False
        self.reward = 0
        self.env.sb_client.ready_for_new_set()
        self.env.sb_client.load_next_available_level()


    def translate_to_observation(self, raw_state):
        processed_state = self.perception.process_state(raw_state)
        return processed_state


    def choose_action(self, observation):
        angle = random.randrange(0, 90, 1)
        return angle

    def store_transition(self, observation, action, reward, observation_, done):
        pass

    def learn(self):
        ### implement learning code here

        pass

    def translate_to_sb_action(self, processed_state, angle_action):
        tp = sb.ScienceBirds.trajectory_planner
        ref_point = tp.get_reference_point(processed_state.sling)
        release_point = tp.find_release_point(processed_state.sling, math.radians(angle_action))
        sb_action = sb.SBShoot(release_point.X, release_point.Y, 3000, ref_point.X, ref_point.Y)
        return sb_action


    def handle_end_of_level(self):
        self.done = True
        self.env.sb_client.load_next_available_level()
        pass

    def handle_novelty_request(self):
        self.env.sb_client.report_novelty_likelihood(0.0, 1.0, [], 0, "")
        self.raw_state = self.env.get_current_state()
        pass

    def handle_evaluation_terminated(self):
        pass


def get_levels_list_for_novelty(novelty, novelty_type):
    samples = 1
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
    novelty_type = 222
    pattern = '9001_Data/StreamingAssets/Levels/novelty_level_{}/type{}/Levels/*.xml'.format(novelty, novelty_type)
    print("running level: " + pattern)
    get_levels_list_for_novelty(novelty, novelty_type)
    env = sb.ScienceBirds(None, launch=True, config=config)
    agent = DQNSBAgent(environment=env)
    agent.main_loop()
    print("all done")

