from agent.gym_hydra_agent import GymHydraAgent
from agent.repair.sb_repair import *

logger = logging.getLogger("repairing_hydra_agent")

from agent.planning.cartpole_pddl_meta_model import *
from agent.repair.cartpole_repair import *


# stats_per_level dictionary keys
REPAIR_TIME = "repair_time"
REPAIR_CALLS = "repair_calls"

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
                                  for i, fluent in enumerate(meta_model_repair.fluents_to_repair)]

            logger.info("Repair done! Repair time %.2f, Consistency: %.2f, Repair:\n %s" % (
            repair_time, consistency, "\n".join(repair_description)))

        super().run(debug_info, max_actions)


