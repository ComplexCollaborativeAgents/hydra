'''
This module runs a set of experiments to evaluate the performance of our ScienceBirds agent
'''
import json
import random
from agent.gym_hydra_agent import *
from runners.run_sb_stats import *

logging.basicConfig(format='%(name)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("sb_experiment_runner")
logger.setLevel(logging.DEBUG)

LOOKUP_PATH = pathlib.Path(__file__).parent.absolute() / "novelty_trials_levels.json"

def load_lookup():
    with open(LOOKUP_PATH) as f:
        obj = json.load(f)
        return obj


''' Run experiments iwth the repairing Hydra agent '''
def run_repairing_sb_experiments():
    # samples = 1000
    # novelties = {0: [2], 1: [6, 7, 8, 9, 10], 2: [6, 7, 8, 9, 10], 3: [6, 7]}
    # novelties = {3: [7]}
    # for agent in [AgentType.Baseline]:
    #    run_performance_stats(novelties, seed=1, agent_type=agent, samples=samples)

    trial_start = 1
    num_trials = 1
    per_trial = 5
    novelties = {22: [1], 23: [1], 24: [1]}
    # novelties = {1: [6, 7, 8, 9, 10]}
    # novelties = {1: [6,], 2:[6,], 3:[6, ]}
    notify_novelty = True
    lookup = load_lookup()

    for agent in [AgentType.Hydra]:
        for trial in range(trial_start, trial_start + num_trials):
            random.seed()
            run_performance_stats(novelties,
                                  agent_type=agent,
                                  samples=per_trial,
                                  suffix=str(trial),
                                  notify_novelty=notify_novelty,
                                  level_lookup=lookup[trial])


if __name__ == '__main__':
    run_repairing_sb_experiments()