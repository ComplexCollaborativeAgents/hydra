import gym
from agent.gym_hydra_agent import GymHydraAgent
from worlds.gym_cartpole_dispatcher import GymCartpoleDispatcher
from worlds.wsu.wsu_dispatcher import WSUObserver


def start_gym_interface():
    env = gym.make("CartPole-v1")
    agent = GymHydraAgent(env)
    agent.run(debug_info=True)


def start_wsu_interface():
    observer = WSUObserver()
    env = GymCartpoleDispatcher(observer)
    env.run()


if __name__ == '__main__':
    start_wsu_interface()
