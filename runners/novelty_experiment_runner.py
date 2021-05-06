from agent.gym_hydra_agent import *
from runners.run_sb_stats import *

'''
M1: Given a set trials with novelty in them, how many does the system detect to be not-novel?
M2: Given a set of trials with novelty in them, how many does the system correctly detect to be novel?
M2.1: Given a set of trial without any novelty, how many does the system incorrectly detect to be novel?
M3: Given a set of trials (novelty and not-novelty), how many trials can the system solve? What does solving mean?
State of art AI player. Average scores for benchmark agents (before and after).
M4: If known that the trial is novel, how many trials can the system solve?
class NoveltyExperimentRunner:
    def __init__(self, environment, agent, novelty_set):    # Requires novelties, the agent, samples per trial, suffix (to differentiate between trials)
        self._env = environment
        self._agent = agent
        pass
    def run_unknown_novelty_experiment(self, novelty_set, samples): ### generate M1, M2, M3
        for i in range(0, samples):
            for novelty in novelty_set:
                for k in range (0, number_of_post_novelty_episode):
                    self._env.set_configuration(novelty)
                    self._agent.initialize() ### initialize the agent to a knowledge state every time a new novelty response is being measured.
                    is_novel, score = self.run(self._env, self._agent, is_novel=?) ### who runs the perceive, decide, act loop?
                    record_data
    def run_known_novelty_experiment(self, novelty_set, samples): ### generate M4
        for i in range(0, samples):
            for novelty in novelty_set:
                self._env.set_configuration(novelty)
                self._agent.initialize()  ### initialize the agent to a knowledge state every time a new novelty response is being measured.
                is_novel, score = self.run(self._env, self._agent, is_novel=True)  ### who runs the perceive, decide, act loop?
                record_data
    def run_no_novelty_experiment(self, samples): ### generate M2.1 and other baseline metrics
        for i in range(0, samples):
            self._env.reset_canonical()
            self._agent.initialize()
            is_novel, score = self.run(self._env, self._agent, is_novel=?)
            record_data
    def run(self):
        is_novel = False
        score = 0
        ### do until success or failure
            ### provide agent with observations
            ### ask the agent to give an action
            ### apply the action to the enivronment and advance to the next timestep and observation
        return is_novel, score
'''

TRIAL_START = 0
NUM_TRIALS = 3
PER_TRIAL = 1
NOVELTIES = {1: [6, 7, 8, 9, 10]}
NOTIFY_NOVELTY  = True

# NOTE: need to change the filename of LOOKUP_PATH to whatever config json file is output by utils/generate_eval_trial_sets
# LOOKUP_PATH = pathlib.Path(__file__).parent.absolute() / "eval_sb_trials_b4_novelty_1.json"
LOOKUP_PATH = pathlib.Path(__file__).parent.absolute() / "eval_sb_trials_short.json"

def load_lookup():
    with open(LOOKUP_PATH) as f:
        obj = json.load(f)
        return obj

if __name__ == "__main__":
    lookup = load_lookup()
    trial_results = []

    for agent in [AgentType.RepairingHydra]:
        for trial_set in range(TRIAL_START, TRIAL_START + NUM_TRIALS):
            random.seed()
            result = run_eval_stats(NOVELTIES,
                                    agent_type=agent,
                                    samples=PER_TRIAL,
                                    suffix=str(trial_set),
                                    notify_novelty=NOTIFY_NOVELTY,
                                    level_lookup=lookup[trial_set])

            trial_results.append(result)


    stat_results = {}

    # M1: avg number of False Negatives among CDTs
    # M2: % of CDTs across all trials

    # Collect CDTs
    CDTs = []
    for results in trial_results:
        for result in results:
            print("STATS: Processing result: {}".format(result))    # Check if trial can be considered CDT
            if result['overall']['true_positives'] <= 0 or result['overall']['false_positives'] != 0:
                break
        else:
            CDTs.append(results)
        print("------------------------------------------------------------------------------------")

    print("STATS: CDTs are: {}".format(CDTs))

    # For every CDT, count false negatives and average
    sum_false_neg = 0
    for cdt in CDTs:
        for result in cdt:
            sum_false_neg += result['overall']['false_negatives']
    if len(CDTs) > 0:
        stat_results['m1'] = sum_false_neg/len(CDTs)
    else:
        stat_results['m1'] = 0

    # Determine % of CDTs
    if len(trial_results) > 0:
        stat_results['m2'] = len(CDTs) / len(trial_results)
    else:
        stat_results['m2'] = 0

    # M2.1: % of Trials with at least 1 False Positive
    # Do 1 - % of CDTs
    stat_results['m2.1'] = 1 - stat_results['m2']

    # M3 + M4: Ratio of agent post-novelty performance vs baseline agent pre-novelty performance (TODO: find pre performance records)

    # M5: Post novelty performance overall vs baseline agent

    # M6: Asymptotic performance vs baseline agent

    # M7: False positive rate and True positive rate    

    print(stat_results)
