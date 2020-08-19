import gym
from agent.gym_hydra_agent import GymHydraAgent

env = gym.make("CartPole-v1")
agent = GymHydraAgent(env)
agent.run()
