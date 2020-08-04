class GymHydraAgent:
    def __init__(self, env):
        self.env = env
        self.observation = self.env.reset()

    def run(self, max_actions=1000):
        for _ in range(max_actions):
            self.env.render()
            action = self.env.action_space.sample()
            self.observation, reward, done, info = self.env.step(action)

            if done:
                self.env.close()
                break
