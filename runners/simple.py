import worlds.science_birds as sb
from agent.hydra_agent import HydraAgent

env = sb.ScienceBirds(None, launch=False)
hydra = HydraAgent(env)
hydra.main_loop()
