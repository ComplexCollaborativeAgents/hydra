from dataclasses import dataclass, field

@dataclass()
class AgentStats:
    """Statistical values tracking the agent's performance
    """
    planning_time: float = 0.0
    explored_states: int = 0

@dataclass
class NoveltyDetectionStats:
    """Statistical values tracking the agent's detection of novelty within an episode
    """
    nn_prob: float = -1.0
    pddl_prob: float = -1.0
    reward_prob: float = -1.0
    novelty_detection: bool = False
    novelty_characterization: dict = {}

# ------------- POLYCRAFT STATS ------------- #

class PolycraftAgentStats(AgentStats):
    """Statistical values tracking the agent's performance in the Polycraft domain
    """
    failed_actions_in_level: int = 0 # Count how many actions have failed in a given level
    actions_since_planning: int = 0 # Count how many actions have been performed since we planned last
    


# ------------- SCIENCE BIRDS STATS ------------- #


# ------------- CARTPOLE STATS ------------- #

