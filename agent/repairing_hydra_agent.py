from agent.hydra_agent import HydraAgent
from agent.consistency.meta_model_repair import *
from agent.gym_hydra_agent import GymHydraAgent

# fh = logging.FileHandler("hydra_repair_debug.log",mode='w')
# formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
# fh.setFormatter(formatter)
# logger = logging.getLogger("hydra_agent")
# logger.setLevel(logging.INFO)
# logger.addHandler(fh)

logging.basicConfig(format='%(name)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("hydra_agent")

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
            meta_model_repair = CartpoleRepair(self.consistency_checker, self.desired_precision)
            repair, _ = meta_model_repair.repair(self.meta_model, observation, delta_t=DEFAULT_DELTA_T)

        super().run(debug_info, max_actions)


''' Repairing Hydra agent for the SB domain '''
class RepairingHydraSBAgent(HydraAgent):
    def __init__(self,env=None):
        super().__init__(env)

        self.consistency_estimator = ScienceBirdsConsistencyEstimator()

        # Create meta_model_repair object
        self.desired_consistency = 25 # The consistency threshold for initiating repair
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
            logger.info("checking for repair...")
            if self.should_repair(last_obs):
                logger.info("Initiating repair...")
                repair, consistency = self.meta_model_repair.repair(self.meta_model, last_obs, delta_t=settings.SB_DELTA_T)
                repair_description = ["Repair %s, %.2f" % (fluent, repair[i])
                                      for i, fluent in enumerate(self.meta_model_repair.fluents_to_repair)]
                logger.info("Repair done! Consistency: %.2f, Repair:\n %s" % (consistency, "\n".join(repair_description)))


        super().handle_game_playing(observation, raw_state)

    ''' Checks if the current model should be repaired'''
    def should_repair(self, observation: ScienceBirdsObservation):
        return check_obs_consistency(observation, self.meta_model, self.consistency_estimator) > self.desired_consistency
