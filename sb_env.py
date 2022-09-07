from agent.hydra_agent import HydraAgent
from agent.sb_hydra_agent import SBHydraAgent
import worlds.science_birds as sb
import numpy


env = sb.ScienceBirds(None, launch=True, config="/home/sailor/hydra/data/science_birds/level-15-novel-bird.xml")
current_level = env.sb_client.load_next_available_level()
raw_state = env.get_current_state()
sb_action = 45.0
raw_state, reward = self.env.act(sb_action)

print(env, raw_state)


