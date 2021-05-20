from agent.gym_hydra_agent import *
from runners.run_sb_stats import *

logging.basicConfig(format='%(name)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("novelty_experiment_runner")

TRIAL_START = 0
NUM_TRIALS = 1
PER_TRIAL = 3
NOVELTIES = {1: [6,7,8,9,10], 2:[6,7,8,9,10], 3:[6,7]}
# NOVELTIES = {1: [6,7,8,9,10],2:}
NOTIFY_NOVELTY  = False

# NOTE: need to change the filename of LOOKUP_PATH to whatever config json file is output by utils/generate_eval_trial_sets
# LOOKUP_PATH = pathlib.Path(__file__).parent.absolute() / "eval_sb_trials_test_full.json"
LOOKUP_PATH = pathlib.Path(__file__).parent.absolute() / "eval_sb_trials_test_short.json"

def load_lookup(lookup_path):
    with open(lookup_path) as f:
        obj = json.load(f)
        return obj

if __name__ == "__main__":
    lookup = load_lookup(LOOKUP_PATH)
    trial_results = []

    for agent in [AgentType.RepairingHydra]:
        for trial_set in range(TRIAL_START, TRIAL_START + NUM_TRIALS):
            random.seed()
            logger.info("Agent {} Commencing Trial {}".format(agent, trial_set))

            result = run_eval_stats(NOVELTIES,
                                    agent_type=agent,
                                    samples=PER_TRIAL,
                                    suffix=str(trial_set),
                                    notify_novelty=NOTIFY_NOVELTY,
                                    level_lookup=lookup[trial_set])

            trial_results.append(result)