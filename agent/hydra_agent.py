import logging
from abc import ABCMeta, abstractmethod
from typing import List
from agent.consistency.consistency_estimator import DomainConsistency
from agent.perception.perception import Perception
from agent.reward_estimation.reward_estimator import RewardEstimator

from agent.repair.meta_model_repair import *
from utils.state import Action, State, World
from utils.stats import AgentStats, NoveltyDetectionStats

# TODO: Maybe push this to the settings file? then every module just adds a logger
logging.basicConfig(format='%(name)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("hydra_agent")
logger.setLevel(logging.DEBUG)

# Flags from ANU
NOVELTY_EXISTENCE_NOT_GIVEN = -1 # The self.novelty_existence value indicating that novelty detection is not given by the environment

# stats_per_level dictionary keys
NN_PROB = "nn_novelty_likelihood"
PDDL_PROB = "pddl_novelty_likelihood"
NOVELTY_LIKELIHOOD = "novelty_likelihood"

class HydraPlanner(metaclass=ABCMeta):
    """A superclass of all the Hydra Planners
    
    Attributes:
        meta_model (MetaModel): MetaModel of the domain
        explored_states (int): tracks the number of states explored
    """
    meta_model: MetaModel
    explored_states: int
    
    def __init__(self, meta_model: MetaModel):
        self.meta_model = meta_model
        self.explored_states = 0

    @abstractmethod
    def make_plan(self, state:State, prob_complexity:int=0) -> List[Action]:
        """The plan should be a list of actions that are executable in the environment 

        Args:
            state (State): World state
            prob_complexity (int, optional): Complexity of the PDDL problem. Defaults to 0.

        Raises:
            NotImplementedError

        Returns:
            List[Action]: sequence of Actions that comprise a plan
        """
        raise NotImplementedError()

    @abstractmethod
    def write_pddl_file(problem: PddlPlusProblem, domain: PddlPlusDomain):
        """Write pddl problem file

        Args:
            problem (PddlPlusProblem): PDDL+ problem object w/ goals, objects, etc.
            domain (PddlPlusDomain): PDDL+ domain object w/ requirements, predicates, functions, constants, etc.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError()

    @abstractmethod
    def extract_actions_from_plan_trace(self, plan_filepath: str) -> List[Action]:
        """Parses the given plan trace file and outputs the plan

        Args:
            plan_filepath (str): path to the pddl plan trace file

        Raises:
            NotImplementedError

        Returns:
            List[Action]: sequence of Actions that comprise a plan
        """
        raise NotImplementedError()

class HydraAgent(metaclass=ABCMeta):
    """ A superclass of all the Hydra agents 
    
    Attributes:
        planner (HydraPlanner)
    """
    # Planning
    planner: HydraPlanner           # Planner object acts as interface between python and pddl planning
    meta_model: MetaModel           # Model of the domain (contains fluents)
    meta_model_repair: RepairModule # Object that repairs the meta model

    # Novelty + consistency detection
    consistency: DomainConsistency
    perception: Perception
    reward_estimator: RewardEstimator
    
    # Stats
    current_stats: AgentStats
    current_detection: NoveltyDetectionStats
    agent_stats: List[AgentStats]                   # List of agent stats per episode
    novelty_detection: List[NoveltyDetectionStats]  # List of novelty detection stats per episode

    # Episode Logs + active plan
    episode_logs: List[HydraEpisodeLog] # List of episode logs/observations
    active_plan: List[Action]           # List of actions that the agent will take


    def __init__(self):
        # Novelty + consistency detection and planning objects should be instantiated by the subclass
        self.episode_logs = []
        self.agent_stats = []
        self.novelty_detection = []

    def novelty_exists(self) -> bool:
        """Helper function to check the log to see if novelty has been detected in

        Returns:
            bool: Whether or not novelty has been detected (logged)
        """
        return self.current_detection.novelty_detection

    def set_novelty_existence(self, exists:bool):
        """Helper function that sets whether or not the agent detected novelty in the current episode

        Args:
            exists (bool): Whether or not novelty exists in this episode (logged to NoveltyDetectionStats)
        """
        self.current_detection.novelty_detection = exists

    def get_agent_stats(self) -> List[AgentStats]:
        """Gets the list of stats objects per episode for the current trial

        Returns:
            List[AgentStats]: List of agent stats per episode
        """
        return self.agent_stats

    @abstractmethod
    def episode_init(self, world: World):
        """Perform setup for the agent at the beginning of an episode

        Raises:
            NotImplementedError
        """
        raise NotImplementedError()

    @abstractmethod
    def episode_end(self) -> NoveltyDetectionStats:
        """Perform cleanup for the agent at the end of an episode
        """
        self.novelty_detection.append(self.current_detection)
        self.agent_stats.append(self.current_stats)
        
        self.current_detection = None
        self.current_stats = None

        return self.novelty_detection[-1]

    @abstractmethod
    def choose_action(self, state: State) -> Action:
        """Choose which action to perform in the given state

        Args:
            state (State): State of the world

        Raises:
            NotImplementedError

        Returns:
            Action: action to take
        """
        raise NotImplementedError()

    @abstractmethod
    def should_repair(self, episode_log: HydraEpisodeLog) -> bool:
        """ Choose if the agent should repair its meta model based on the given episode log

        Args:
            episode_log (HydraEpisodeLog): episode log to examine

        Raises:
            NotImplementedError

        Returns:
            bool: whether or not a repair should be initiated
        """
        raise NotImplementedError()

    @abstractmethod
    def repair_meta_model(self, episode_log: HydraEpisodeLog):
        """Call the repair object to repair the current meta model

        Args:
            episode_log (HydraEpisodeLog): episode log to use for repairs

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError()

    @abstractmethod
    def _detect_novelty(self, episode_log: HydraEpisodeLog) -> bool:
        """Given an episode log, determine using the methods available (consistency, perception, reward_prediction, etc) whether or not novelty is present

        Args:
            episode_log (HydraEpisodeLog):  episode log to examine

        Raises:
            NotImplementedError

        Returns:
            bool: Whether or not novelty was detected
        """

        raise NotImplementedError()

    @abstractmethod
    def _build_episode_log(self, state: State) -> HydraEpisodeLog:
        """_summary_

        Args:
            state (State): State of the world

        Raises:
            NotImplementedError: _description_

        Returns:
            HydraEpisodeLog: _description_
        """

        raise NotImplementedError()

