from agent.sb_hydra_agent import SBHydraAgent


def test_create_agent():
    h = SBHydraAgent()
    assert isinstance(h, SBHydraAgent)
