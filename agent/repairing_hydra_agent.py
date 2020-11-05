from agent.consistency.sb_repair import ScienceBirdsMetaModelRepair
from agent.hydra_agent import HydraAgent
from agent.consistency.meta_model_repair import *
from agent.gym_hydra_agent import GymHydraAgent
import os.path as path

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
        super().__init__(env, agent_stats)
        self.debug_mode = True
        self.meta_model_repair = ScienceBirdsMetaModelRepair()

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
                logger.info("Initiating repair...")
                if self.debug_mode: # Print observation and meta model to allow debugging
                    obs_output_file = path.join(settings.ROOT_PATH, "data", "science_birds", "current_repair.p")  # For debug
                    pickle.dump(last_obs, open(obs_output_file, "wb"))  # For debug
                    obs_output_file = path.join(settings.ROOT_PATH, "data", "science_birds",
                                                "current_repair.mm")  # For debug
                    pickle.dump(self.meta_model, open(obs_output_file, "wb"))  # For debug

                start_time = time.time()
                repair, consistency = self.meta_model_repair.repair(self.meta_model,
                                                                    last_obs,
                                                                    delta_t=settings.SB_DELTA_T)
                repair_time = time.time()-start_time

                self.stats_for_level["repair_time"] = repair_time
                self.stats_for_level["repair_called"] = 1
                self.stats_for_level["consistency_after_repair"] = consistency

                repair_description = ["Repair %s, %.2f" % (fluent, repair[i])
                                      for i, fluent in enumerate(self.meta_model_repair.fluents_to_repair) if repair[i]!=0]
                logger.info("Repair done! Consistency: %.2f, Runtime %.2f, Repaired:\n %s" %
                            (consistency, repair_time, "\n".join(repair_description)))


        super().handle_game_playing(observation, raw_state)

    ''' Checks if the current model should be repaired'''
    def should_repair(self, observation: ScienceBirdsObservation):
        return check_obs_consistency(observation, self.meta_model, self.meta_model_repair.consistency_estimator, simulator=RefinedPddlPlusSimulator()) \
               > self.meta_model_repair.consistency_threshold
