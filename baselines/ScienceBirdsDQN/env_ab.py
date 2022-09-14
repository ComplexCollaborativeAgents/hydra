import logging, pathlib
import math, cv2
import settings
import worlds.science_birds as sb
from worlds.science_birds_interface.client.agent_client import GameState
from agent.perception.perception import Perception
from agent.reward_estimation.nn_utils.obs_to_imgs import SBObs_to_Imgs
import random
from runners.run_sb_stats import prepare_config
import numpy as np
SB_DATA_PATH = pathlib.Path(settings.ROOT_PATH) / 'data' / 'science_birds'
SB_CONFIG_PATH = SB_DATA_PATH / 'config'
SB_BIN_PATH = pathlib.Path(settings.SCIENCE_BIRDS_BIN_DIR) / 'linux'

from matplotlib import pyplot as plt


class Env():
    def __init__(self, environment):
        self.log = logging.getLogger(__name__).getChild('DQNSBAgent')
        self.env = environment
        self.perception = Perception()
        self.state_coverter = SBObs_to_Imgs()
        pass


    def reset(self):
        token = True
        while token:
            raw_state = self.env.get_current_state()
            #self.handle_new_trial()
            if raw_state.game_state.value == GameState.NEWTRIAL.value:
                self.handle_new_trial()
            if raw_state.game_state.value == GameState.NEWTRAININGSET.value:
                self.handle_new_training_set()           
            if raw_state.game_state.value == GameState.REQUESTNOVELTYLIKELIHOOD.value:
                self.handle_novelty_request()
            if raw_state.game_state.value == GameState.PLAYING.value:
                processed_state = self.perception.process_state(raw_state)
                image_state = self.state_coverter.state_to_nD_img(processed_state.objects)
                token = False
        return image_state
        
    #observation_, reward, done, info = env.step(action)
    def step(self, action_):
        token = True
        done = False
        while token:
            raw_state = self.env.get_current_state()
            if raw_state.game_state.value == GameState.NEWTRIAL.value:
                self.handle_new_trial()
            
            if raw_state.game_state.value == GameState.NEWTRAININGSET.value:
                self.handle_new_training_set()

            if raw_state.game_state.value == GameState.PLAYING.value:
                
                processed_state = self.perception.process_state(raw_state)
                action = self.generate_sb_action(processed_state, action_) 
                raw_state, reward = self.env.act(action)                
                image_state = self.state_coverter.state_to_nD_img(processed_state.objects)
                token = False
            if raw_state.game_state.value == GameState.REQUESTNOVELTYLIKELIHOOD.value:
                self.handle_novelty_request()

            if raw_state.game_state.value == GameState.LOST.value:
                self.handle_game_lost()
                done = True
            if raw_state.game_state.value == GameState.WON.value:
                self.handle_game_won()
                done = True 
        return image_state, reward, done, None


    def handle_new_trial(self):
        ## reset agent
        self.env.sb_client.ready_for_new_set()
        self.env.sb_client.load_next_available_level()

    def handle_new_training_set(self):
        self.env.sb_client.ready_for_new_set()
        self.env.sb_client.load_next_available_level()


    def generate_sb_action(self, processed_state, angle):
        tp = sb.ScienceBirds.trajectory_planner
        ref_point = tp.get_reference_point(processed_state.sling)
        release_point = tp.find_release_point(processed_state.sling, math.radians(angle))
        sb_action = sb.SBShoot(release_point.X, release_point.Y, 3000, ref_point.X, ref_point.Y)
        return sb_action


    def handle_game_lost(self):
        self.env.sb_client.load_next_available_level()
        pass

    def handle_game_won(self):
        self.env.sb_client.load_next_available_level()
        pass

    def handle_novelty_request(self):
        self.env.sb_client.report_novelty_likelihood(0.0, 1.0, [], 0, "")
        pass


def get_levels_list_for_novelty(novelty, novelty_type, config):
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
    

#env = gym.make('LunarLander-v2')
def make_sb():
    config = SB_CONFIG_PATH/'stats_config.xml'
    novelty = 0
    novelty_type = 222
    pattern = '9001_Data/StreamingAssets/Levels/novelty_level_{}/type{}/Levels/*.xml'.format(novelty, novelty_type)
    print("running level: " + pattern)
    get_levels_list_for_novelty(novelty, novelty_type, config)
    env = sb.ScienceBirds(None, launch=True, config=config)
    env_sb = Env(environment=env)
    return env_sb

"""     
if __name__ == '__main__':
    env = make_sb()
    observation = env.reset()
    print("step1\n\n\n", np.shape(observation))
    action = 40.0
    observation_, reward, done, info = env.step(action)
    print("step2\n\n\n", np.shape(observation), np.shape(observation_), reward)
"""
