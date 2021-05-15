from agent.hydra_agent import HydraAgent
from agent.repair.meta_model_repair import *
from agent.gym_hydra_agent import GymHydraAgent
import os.path as path
from state_prediction.anomaly_detector_fc_multichannel import FocusedSBAnomalyDetector
from agent.repair.sb_repair import *


logger = logging.getLogger("repairing_hydra_agent")

from agent.planning.cartpole_pddl_meta_model import *
from agent.repair.cartpole_repair import *


# Flags from ANU
NOVELTY_EXISTANCE_NOT_GIVEN = -1 # The self.novelty_existance value indicating that novelty detection is not given by the environment

# stats_per_level dictionary keys
NN_PROB = "nn_novelty_likelihood"
PDDL_PROB = "pddl_novelty_likelihood"
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
        self.consistency_estimator = ScienceBirdsConsistencyEstimator()
        self.detector = FocusedSBAnomalyDetector()

        # Repair and detection variables
        self.revision_attempts = 0
        self.nn_prob_per_level = []
        self.meta_model_repair = ScienceBirdsMetaModelRepair(self.meta_model)



    def reinit(self):
        super().reinit()
        self.revision_attempts = 0
        self.nn_prob_per_level = []


    def process_final_observation(self):
        ''' This is called after winning or losing a level. '''
        last_obs = self.find_last_obs()

        self._compute_novelty_likelihood(last_obs)
        self.stats_for_level["novelty_likelihood"]=self.novelty_likelihood
        # The consistency score per level for this level is the mean over the consistency scored of this level's observations
        self.nn_prob_per_level.insert(0,
                                      sum(self.stats_for_level[NN_PROB]) / len(self.stats_for_level[NN_PROB]))


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

    def _compute_novelty_likelihood(self, observation: ScienceBirdsObservation):
        ''' Computes the novelty likelihood score.
        Also updates the stats_for_level object with the computed novelty probability by the two models.  '''

        if NN_PROB not in self.stats_for_level:
            self.stats_for_level[NN_PROB]=[]
        if PDDL_PROB not in self.stats_for_level:
            self.stats_for_level[PDDL_PROB]=[]

        # if novelty existences is given by the experiment framework - no need to run the fancy models
        if self.novelty_existence in  [0,1]:
            self.stats_for_level[NN_PROB].append(self.novelty_existence)
            self.stats_for_level[PDDL_PROB].append(self.novelty_existence)
            self.novelty_likelihood = self.novelty_existence
        else:
            assert self.novelty_existence==NOVELTY_EXISTANCE_NOT_GIVEN # The flag denoting that we do not get novelty info from the environment

            if observation.hasUnknownObj():
                self.stats_for_level[NN_PROB].append(1.0)
                self.stats_for_level[PDDL_PROB].append(1.0)
                self.novelty_likelihood = 1.0
            else:
                try:
                    cnn_novelty, cnn_prob = self.detector.detect(observation)
                except:
                    logging.info('CNN Index out of Bounds in game playing')
                    cnn_prob=1.0 # TODO: Think about this design choice

                self.stats_for_level[NN_PROB].append(cnn_prob)

                if settings.NO_PDDL_CONSISTENCY:
                    pddl_prob = 1.0
                else:
                    pddl_prob = check_obs_consistency(observation, self.meta_model, self.consistency_estimator)
                self.stats_for_level["pddl_novelty_likelihood"].append(pddl_prob)

                # If we already played at least two levels and novelty keeps being detected, mark this as a very high novelty likelihood
                if cnn_prob > self.detector.threshold and \
                    pddl_prob > self.meta_model_repair.consistency_threshold and \
                    len(self.completed_levels)>1 and \
                    not self.completed_levels[-1] and \
                        self.nn_prob_per_level[0] > self.detector.threshold and \
                        self.nn_prob_per_level[1] > self.detector.threshold:
                    self.novelty_likelihood = 1.0
                else:
                    self.novelty_likelihood = cnn_prob

        # Record current novelty likelihood estimate
        self.stats_for_level["novelty_likelihood"]=self.novelty_likelihood


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
        # If novelty existance is given, use the given value
        if self.novelty_existence != NOVELTY_EXISTANCE_NOT_GIVEN:
            return self.novelty_existence==1

        # Compute novelty likelihood values (cnn prob, pddl prob, and novelty likelihood)
        self._compute_novelty_likelihood(observation)

        if observation.hasUnknownObj():
            return True

        cnn_prob = self.stats_for_level[NN_PROB][-1]
        pddl_prob = self.stats_for_level[PDDL_PROB][-1]

        # Try to repair only after 2 levels have passed & only after the first shot of the level # TODO: Rethink this design choice
        if len(self.completed_levels) < 2 or len(self.stats_for_level[NN_PROB]) != 1:
            return False

        logger.info("CNN novelty likelihoods last shot: %.3f, previous problem: %.3f, two problems ago: %.3f, last problem solved? %s" % (cnn_prob,
                                                                                                                                          self.nn_prob_per_level[0],
                                                                                                                                          self.nn_prob_per_level[1],
                                                                                                                                          self.completed_levels[-1]))

        # Only repair if the following conditions hold
        if self.revision_attempts < settings.HYDRA_MODEL_REVISION_ATTEMPTS and\
                cnn_prob > self.detector.threshold and\
                self.nn_prob_per_level[0] > self.detector.threshold and\
                self.nn_prob_per_level[1] > self.detector.threshold and\
                not self.completed_levels[-1] and \
                pddl_prob > self.meta_model_repair.consistency_threshold:
            return True

        return False