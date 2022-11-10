import datetime
from abc import ABCMeta, abstractmethod
from typing import Dict

from utils.state import World
from agent.hydra_agent import HydraAgent
from utils.state import World

class Dispatcher(metaclass=ABCMeta):
    """ SuperClass for hydra agent dispatcher. Interfaces connecting hydra
            agnet and environment for running an evaluation.

    Attributes:
    """
    world: World                # trial/evaluation that is being run
    agent: HydraAgent           # hydra agent being tested
    trial_timestamp: str

    def __init__(self, agent:HydraAgent):
        self.trial_timestamp = datetime.datetime.now().strftime("%y%m%d%H%M%S")
        self.agent = agent


    @abstractmethod
    def report_novelty(self) -> dict:
        """ Return a dictionary with novelty detail and stats

        Returns:
            Dict[str, Any]: Dictionary with novelty details

        Raises:
            NotImplementedError
        """
        raise NotImplementedError()


    @abstractmethod
    def run(self):
        """ Run an evaluation for self.agent in self.world

        Raises:
            NotImplementedError
        """
        raise NotImplementedError()


    @abstractmethod
    def run_experiment(self):
        """ Run an experiment

        Raises:
            NotImplementedError
        """
        raise NotImplementedError()

    @abstractmethod
    def cleanup_experiment(self):
        """ Perform cleanup after running the experiment

        Raises:
            NotImplementedError
        """
        raise NotImplementedError()

    @abstractmethod
    def run_trial(self):
        """ Run a trial for the experiment

        Raises:
            NotImplementedError
        """
        raise NotImplementedError()

    @abstractmethod
    def cleanup_trial(self):
        """ Perform clean after running an experiment.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError()

    @abstractmethod
    def run_episode(self):
        """ Run one episode of the trial

        Raises:
            NotImplementedError
        """
        raise NotImplementedError()

    @abstractmethod
    def cleanup_episode(self):
        """ Perform cleanup after running an episode of a trial

        Raises:
            NotImplementedError
        """
        raise NotImplementedError()


