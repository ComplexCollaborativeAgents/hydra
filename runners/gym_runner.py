import gym

from agent.cartpole_hydra_agent import CartpoleHydraAgentObserver, RepairingCartpoleHydraAgent
from agent.gym_hydra_agent import GymHydraAgent
from worlds.gym_cartpole_dispatcher import GymCartpoleDispatcher


def start_gym_interface():
    env = gym.make("CartPole-v1")
    agent = GymHydraAgent(env)
    agent.run(debug_info=True)


def start_wsu_interface():
    observer = CartpoleHydraAgentObserver(agent_type=RepairingCartpoleHydraAgent)
    env = GymCartpoleDispatcher(observer, render=True)
    env.run()


if __name__ == '__main__':
    start_wsu_interface()
