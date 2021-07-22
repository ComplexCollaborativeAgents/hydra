from agent.sb_hydra_agent import RepairingSBHydraAgent
from agent.planning.sb_meta_model import *
import worlds.science_birds as sb


def test_science_birds_agent():
    env = sb.ScienceBirds(None,launch=True,config='test_reinit.xml')
    settings.HYDRA_MODEL_REVISION_ATTEMPTS = 0
    hydra = RepairingSBHydraAgent(env)
    metamodel = hydra.meta_model
    hydra.main_loop() # enough actions to play the first two levels
    env.kill()
    assert(hydra.meta_model is not metamodel)
    assert(hydra.planner.meta_model is not metamodel)
    assert(len(hydra.env.history) == 1) # 1 shot not 2

