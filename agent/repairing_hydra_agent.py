from agent.hydra_agent import HydraAgent
from agent.consistency.meta_model_repair import *
from agent.gym_hydra_agent import GymHydraAgent
import os.path as path
from state_prediction.anomaly_detector import FocusedSBAnomalyDetector
from agent.consistency.sb_repair import *

fh = logging.FileHandler("hydra.log",mode='w')
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger = logging.getLogger("hydra_agent")
logger.setLevel(logging.INFO)
logger.addHandler(fh)

from agent.planning.cartpole_pddl_meta_model import *
from agent.consistency.cartpole_repair import *

class RepairingGymHydraAgent(GymHydraAgent):
    def __init__(self, env, starting_seed=False):
        super().__init__(env, starting_seed)
        self.consistency_checker = CartpoleConsistencyEstimator()
        self.desired_precision = 0.01

    ''' Checks if the meta model should be repaired based on the given observation. Note: can also consider past observations'''
    def should_repair(self, observation):
        if sum(observation.rewards)>195:
            return False
        else:
            return True

    def run(self, debug_info=False, max_actions=1000):
        observation = self.find_last_obs()
        if observation is not None:
            # Initiate repair
            if not self.should_repair(observation):
                logger.info("No need to repair")
                return
            meta_model_repair = CartpoleRepair()
            start_time = time.time()
            repair, consistency = meta_model_repair.repair(self.meta_model, observation, delta_t=settings.CP_DELTA_T)
            repair_time = time.time()-start_time
            repair_description = ["Repair %s, %.2f" % (fluent, repair[i])
                                  for i, fluent in enumerate(self.meta_model_repair.fluents_to_repair)]

            logger.info("Repair done! Repair time %.2f, Consistency: %.2f, Repair:\n %s" % (
            repair_time, consistency, "\n".join(repair_description)))

        super().run(debug_info, max_actions)


''' Repairing Hydra agent for the SB domain '''
class RepairingHydraSBAgent(HydraAgent):
    def __init__(self,env=None,agent_stats = list()):
        super().__init__(env)
        self.consistency_estimator = ScienceBirdsConsistencyEstimator()
        self.detector = FocusedSBAnomalyDetector()

        # Create meta_model_repair object
        self.revision_attempts = 0
        self.meta_model_repair = ScienceBirdsMetaModelRepair(self.meta_model)

    def reinit(self):
        super().reinit()
        self.revision_attempts = 0

    ''' Handle what happens when the agent receives a PLAYING request'''
    def handle_game_playing(self, observation, raw_state):
        last_obs = self.find_last_obs()
        if last_obs!=None:
            if "repair_calls" not in self.stats_for_level:
                self.stats_for_level["repair_called"] = 0
            if "repair_time" not in self.stats_for_level:
                self.stats_for_level["repair_time"] = 0

            # Check if we should repair
            logger.info("checking for repair...")
            if self.should_repair(last_obs):
                self.revision_attempts += 1
                logger.info("Initiating repair number {}".format(self.revision_attempts))
                repair, consistency = self.meta_model_repair.repair(self.meta_model, last_obs)
                repair_description = ["Repair %s, %.2f" % (fluent, repair[i])
                                      for i, fluent in enumerate(self.meta_model_repair.fluents_to_repair)]
                logger.info("Repair done! Consistency: %.2f, Repair:\n %s" % (consistency, "\n".join(repair_description)))


        super().handle_game_playing(observation, raw_state)


    ''' Checks if the current model should be repaired'''
    def should_repair(self, observation: ScienceBirdsObservation):
        # novelty existences should be -1, 0, 1 but we are still waiting on Peng
        if (self.novelty_existence == 0) or self.revision_attempts >= settings.HYDRA_MODEL_REVISION_ATTEMPTS:
            return False
        elif self.novelty_existence == 1:
            return True
        elif self.novelty_existence == -1 and \
                (self.completed_levels and self.completed_levels[-1] == False):
            if observation.hasUnknownObj():
                self.novelty_likelihood = 1
                return True
            cnn_novelty, cnn_prob = self.detector.detect(observation)
            self.novelty_likelihood = max(self.novelty_likelihood, cnn_prob)
            return cnn_novelty and check_obs_consistency(observation, self.meta_model, self.consistency_estimator,
                                                         simulator=RefinedPddlPlusSimulator(),
                                                         delta_t=settings.SB_DELTA_T) > self.meta_model_repair.consistency_threshold
        else:
            return False