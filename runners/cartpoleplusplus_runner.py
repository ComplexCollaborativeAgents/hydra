import logging

import gym

from worlds.cartpoleplusplus_dispatcher import CartPolePlusPlusDispatcher
from worlds.wsu.generator.m_1 import CartPolePPMock1
from worlds.wsu.generator.m_2 import CartPolePPMock2
from worlds.wsu.generator.n_0 import CartPole
from worlds.wsu.generator.n import CartPoleNoBlocks

from worlds.wsu.wsu_dispatcher import WSUObserver
from agent.cartpoleplusplus_hydra_agent import CartpolePlusPlusHydraAgentObserver, RepairingCartpolePlusPlusHydraAgent, CartpolePlusPlusHydraAgent


def start_wsu_interface():
    observer = CartpolePlusPlusHydraAgentObserver(RepairingCartpolePlusPlusHydraAgent)
    # observer = WSUObserver()
    env = CartPolePlusPlusDispatcher(observer, render=True)
    env.run(generators=[CartPole], difficulties=['easy', 'medium', 'hard'])


def setup_logging(level: int = logging.INFO):
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)

    # disable gym warnings:
    gym.logger.setLevel(gym.logger.ERROR)


if __name__ == '__main__':
    setup_logging()
    start_wsu_interface()
