from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass()
class AgentStats:
    """Statistical values tracking the agent's performance
    """
    episode_start_time: float
    planning_time: float = 0.0
    explored_states: int = 0
    repair_calls: int = 0
    repair_time: float = 0.0
    plan_action_length: int = 0
    timed_out: bool = False
    success: bool = False

@dataclass
class NoveltyDetectionStats:
    """Statistical values tracking the agent's detection of novelty within an episode
    """
    nn_prob: float = -1.0
    pddl_prob: float = -1.0
    reward_prob: float = -1.0
    novelty_detection: bool = False
    novelty_characterization: dict = field(default_factory=dict)    # Dict[str, Any]

# ------------- POLYCRAFT STATS ------------- #

class PolycraftAgentStats(AgentStats):
    """Statistical values tracking the agent's performance in the Polycraft domain
    """
    failed_actions: int = 0 # Count how many actions have failed in a given level
    actions_since_planning: int = 0 # Count how many actions have been performed since we planned last
    
class PolycraftDetectionStats(NoveltyDetectionStats):
    """Statistical values tracking the agent's detection of novelty within an episode of the Polycraft Domain
    """
    

# ------------- SCIENCE BIRDS STATS ------------- #


# ------------- CARTPOLE STATS ------------- #

