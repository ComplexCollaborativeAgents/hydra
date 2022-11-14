from dataclasses import dataclass, field
from typing import Dict, Any, List

@dataclass
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
    novelty_detected: bool = False
    novelty_characterization: dict = field(default_factory=dict)    # Dict[str, Any]

# ------------- POLYCRAFT STATS ------------- #

@dataclass
class PolycraftAgentStats(AgentStats):
    """Statistical values tracking the agent's performance in the Polycraft domain
    """
    failed_actions: int = 0 # Count how many actions have failed in a given level
    actions_since_planning: int = 0 # Count how many actions have been performed since we planned last
    
@dataclass
class PolycraftDetectionStats(NoveltyDetectionStats):
    """Statistical values tracking the agent's detection of novelty within an episode of the Polycraft Domain
    """
    

# ------------- SCIENCE BIRDS STATS ------------- #

@dataclass
class SBAgentStats(AgentStats):
    """"""
    default_action_used: bool = False
    num_objects: int = 0
    shot_count: int = 0
    plan_total_time: float = 0.0
    cumulative_plan_time: float = 0.0
    rewards_per_shot: List[float] = field(default_factory=list)
    repair_description: List[List[str]] = field(default_factory=list) # List of lists of repairs

@dataclass
class SBDetectionStats(NoveltyDetectionStats):
    """"""
    unknown_obj: bool = False

# ------------- CARTPOLE STATS ------------- #

