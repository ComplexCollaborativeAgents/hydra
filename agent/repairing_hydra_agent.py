import agent.hydra_agent as hydra_agent
from agent.hydra_agent import HydraAgent
from agent.repair.meta_model_repair import *
from agent.gym_hydra_agent import GymHydraAgent
import os.path as path
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


''' Repairing Hydra agent for the SB domain '''
class RepairingHydraSBAgent(HydraAgent):
    def __init__(self,env=None, agent_stats = list()):
        super().__init__(env, agent_stats=agent_stats)
        # Repair and detection variables
        self.revision_attempts = 0
        self.meta_model_repair = ScienceBirdsMetaModelRepair(self.meta_model)



    def reinit(self):
        super().reinit()
        self.revision_attempts = 0



    def process_final_observation(self):
        ''' This is called after winning or losing a level. '''
        self.stats_for_level[hydra_agent.NOVELTY_LIKELIHOOD]=self.novelty_likelihood
        # The consistency score per level for this level is the mean over the consistency scored of this level's observations
        self.nn_prob_per_level.insert(0,
                                      sum(self.stats_for_level[hydra_agent.NN_PROB]) / len(self.stats_for_level[hydra_agent.NN_PROB]))
        self.pddl_prob_per_level.insert(0,
                                      sum(self.stats_for_level[hydra_agent.PDDL_PROB]) / len(self.stats_for_level[hydra_agent.PDDL_PROB]))

    def handle_evaluation_terminated(self):
        ''' Handle what happens when the agent receives a EVALUATION_TERMINATED request'''
        self.process_final_observation()
        return super().handle_evaluation_terminated()

    def handle_game_won(self):
        self.process_final_observation()
        super().handle_game_won()

    def handle_game_lost(self):
        self.process_final_observation()
        super().handle_game_lost()

    ''' Handle what happens when the agent receives a PLAYING request'''
    def handle_game_playing(self, observation, raw_state):
        last_obs = self.find_last_obs()
        if last_obs!=None:
            if REPAIR_CALLS not in self.stats_for_level:
                self.stats_for_level[REPAIR_CALLS] = 0
            if REPAIR_TIME not in self.stats_for_level:
                self.stats_for_level[REPAIR_TIME] = 0

            # Check if we should repair
            logger.info("checking for repair...")
            if self.should_repair(last_obs) and settings.NO_REPAIR==False:
                self.stats_for_level[REPAIR_CALLS] = self.stats_for_level[REPAIR_CALLS] + 1
                self.revision_attempts += 1
                logger.info("Initiating repair number {}".format(self.revision_attempts))

                start_repair_time = time.time()
                try:
                    repair, consistency = self.meta_model_repair.repair(self.meta_model, last_obs)
                    repair_description = ["Repair %s, %.2f" % (fluent, repair[i])
                                          for i, fluent in enumerate(self.meta_model_repair.fluents_to_repair)]
                    logger.info(
                        "Repair done! Consistency: %.2f, Repair:\n %s" % (consistency, "\n".join(repair_description)))
                except:
                    # TODO: fix this hack
                    logger.info("Repair failed!")
                self.stats_for_level[REPAIR_TIME] = time.time() - start_repair_time

        # In should_repair, self.novelty_likelihood is turned into a 1 if past 3 consistency scores are high enough (and is otherwise < 1).  
        # 1 should be considered True, anything else should be considered False
        logger.info("Novelty likelihood is {}".format(self.novelty_likelihood))

        super().handle_game_playing(observation, raw_state)


    '''
    Checks if the current model should be repaired
    If we are going to repair for a level, it will be a repair with the first shot's observations for that level.
    '''
    def should_repair(self, observation: ScienceBirdsObservation):
        # If novelty existance is given, use the given
        if self.revision_attempts >= settings.HYDRA_MODEL_REVISION_ATTEMPTS:
            return False

        if self.novelty_existence != hydra_agent.NOVELTY_EXISTANCE_NOT_GIVEN:
            return self.novelty_existence==1

        if observation.hasUnknownObj():
            return True

        if hydra_agent.NN_PROB not in self.stats_for_level:
            return False #TODO: Design choice: wait for the second shot to repair

        cnn_prob = self.stats_for_level[hydra_agent.NN_PROB][-1]
        pddl_prob = self.stats_for_level[hydra_agent.PDDL_PROB][-1]

        # Try to repair only after 2 levels have passed & only after the first shot of the level # TODO: Rethink this design choice
        if len(self.completed_levels) < 2 or len(self.stats_for_level[hydra_agent.NN_PROB]) != 1:
            return False

        logger.info("CNN novelty likelihoods last shot: %.3f, previous problem: %.3f, two problems ago: %.3f, last problem solved? %s" % (cnn_prob,
                                                                                                                                          self.nn_prob_per_level[0],
                                                                                                                                          self.nn_prob_per_level[1],
                                                                                                                                          self.completed_levels[-1]))

        # Only repair if the following conditions hold
        if cnn_prob > self.detector.threshold and\
                self.nn_prob_per_level[0] > self.detector.threshold and\
                self.nn_prob_per_level[1] > self.detector.threshold and\
                not self.completed_levels[-1] and \
                pddl_prob > self.meta_model_repair.consistency_threshold:
            return True

        return False