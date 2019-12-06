from agent.hydra_agent import HydraAgent

def test_create_agent():
    h = HydraAgent()
    assert isinstance(h, HydraAgent)
