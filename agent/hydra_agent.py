import logging

from agent.repair.meta_model_repair import *

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

class HydraPlanner:
    """ A superclass of all the Hydra Planners"""
    def __init__(self, meta_model: MetaModel):
        self.meta_model = meta_model
        self.explored_states = 0


    def make_plan(self,state,prob_complexity=0):
        """ The plan should be a list of actions that are executable in the environment """
        raise NotImplementedError()


class HydraAgent:
    """ A superclass of all the Hydra agents """
    def __init__(self, planner : HydraPlanner,
                 meta_model_repair : MetaModelRepair):
        if planner is not None:
            self.meta_model = planner.meta_model
        self.planner = planner
        self.meta_model_repair = meta_model_repair
        self.observations_list = []
        self.novelty_likelihood = 0.0 # An estimate of the likelihood that novelty has been introduced
        self.novelty_existence = -1 # A flag indicating whether we received an explicit message from the environment that novelty has bene introduced

    def choose_action(self, world_state):
        """ Choose which action to perform in the given state """
        raise NotImplementedError()

    def should_repair(self, observation):
        """ Choose if the agent should repair its meta model based on the given observation """
        raise NotImplementedError()

    def repair_meta_model(self, observation):
        """ Call the repair object to repair the current meta model """
        raise NotImplementedError()

