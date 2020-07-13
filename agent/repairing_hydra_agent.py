from agent.hydra_agent import HydraAgent
from agent.consistency.meta_model_repair import *
from agent.consistency.consistency_estimator import check_obs_consistency, BirdLocationConsistencyEstimator
fh = logging.FileHandler("hydra.log",mode='w')
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger = logging.getLogger("hydra_agent")
logger.setLevel(logging.INFO)
logger.addHandler(fh)

class RepairingHydraAgent(HydraAgent):
    '''
    Probably needs to subclass for each domain. We will cross that bridge when we get there
    '''

    def __init__(self,env=None):
        super(RepairingHydraAgent, self).__init__(env)

        self.consistency_estimator = BirdLocationConsistencyEstimator()

        # Create meta_model_repair object
        self.desired_consistency = 10 # The consistency threshold for initiating repair
        constants_to_repair = list(self.meta_model.repairable_constants)
        repair_deltas = [1.0] * len(constants_to_repair)
        self.meta_model_repair = GreedyBestFirstSearchMetaModelRepair(constants_to_repair,
                                                                      self.consistency_estimator,
                                                                      repair_deltas,
                                                                      consistency_threshold=self.desired_consistency)

    ''' Handle what happens when the agent receives a PLAYING request'''
    def handle_game_playing(self, observation, raw_state):
        last_obs = self.find_last_obs()
        if last_obs!=None:
            # Check if we should repair
            if self.should_repair(last_obs):
                logger.info("Initiating repair...")
                repair, consistency = self.meta_model_repair.repair(self.meta_model, last_obs)
                repair_description = ["Repair %s, %.2f" % (fluent, repair[i])
                                      for i, fluent in enumerate(self.meta_model_repair.fluents_to_repair)]
                logger.info("Repair done! Consistency: %.2f, Repair:\n %s" % (consistency, "\n".join(repair_description)))


        super().handle_game_playing(observation, raw_state)

    ''' Checks if the current model should be repaired'''
    def should_repair(self, observation: ScienceBirdsObservation):
        return check_obs_consistency(observation, self.meta_model, self.consistency_estimator) > self.desired_consistency
