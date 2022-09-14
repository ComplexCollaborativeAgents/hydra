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
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class QNet():
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


class DQNSBAgent():
    def __init__(self, environment):
        self.log = logging.getLogger(__name__).getChild('DQNSBAgent')
        self.env = environment
        self.perception = Perception()
        self.state_coverter = SBObs_to_Imgs()
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
                print("reward", reward)
            if raw_state.game_state.value == GameState.REQUESTNOVELTYLIKELIHOOD.value:
                self.handle_novelty_request()

            if raw_state.game_state.value == GameState.LOST.value:
                self.handle_game_lost()
                print("done")
            if raw_state.game_state.value == GameState.WON.value:
                self.handle_game_won()
                print("done")


    def handle_new_trial(self):
        ## reset agent
        self.env.sb_client.ready_for_new_set()
        self.env.sb_client.load_next_available_level()

    def handle_new_training_set(self):
        self.env.sb_client.ready_for_new_set()
        self.env.sb_client.load_next_available_level()

    def choose_action(self, raw_state):
        processed_state = self.perception.process_state(raw_state)
        image_state = self.state_coverter.state_to_nD_img(processed_state.objects)
        print (np.shape(image_state))
        #### debug/visualize state generation
        #image_pic = self.state_coverter.state_to_image(processed_state.objects)
        #plt.imshow(image_pic, interpolation='nearest')
        #plt.show()
        #### code to choose the right action by sampling the QNet

        angle = random.randrange(0, 90, 1)
        print("action:", angle)

        ####
        action = self.generate_sb_action(processed_state, angle)
        return action

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
    novelty_type = 222
    pattern = '9001_Data/StreamingAssets/Levels/novelty_level_{}/type{}/Levels/*.xml'.format(novelty, novelty_type)
    print("running level: " + pattern)
    get_levels_list_for_novelty(novelty, novelty_type)
    env = sb.ScienceBirds(None, launch=True, config=config)
    agent = DQNSBAgent(environment=env)
    agent.main_loop()

