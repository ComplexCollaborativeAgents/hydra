import worlds.room_world as rm
import agent.hydra_agent as ha
import numpy as np

np.random.seed(0)

env = rm.RoomWorld()  # initialize world
agent = ha.HydraAgent(env)

# Training
total_episodes = 100
max_steps = 20
for i in range(total_episodes):
    print("level %s started" % i)
    agent.set_env(rm.RoomWorld()) #reset env
    agent.main_loop() # complete the level, maybe this should be part of the planner as well
    print("level %s done" % i)

print("Trained Q matrix:")
print(agent.rl.Q / np.max(agent.rl.Q) * 100)

agent.set_env(rm.RoomWorld())  # reset env
agent.rl.epsilon = 0 # should just do policy and not doing learning
agent.main_loop()
print("Plan: " + str(agent.env.history))
assert(agent.env.history == [5,6,7])
