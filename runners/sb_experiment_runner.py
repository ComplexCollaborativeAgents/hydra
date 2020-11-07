'''
This module runs a set of experiments to evaluate the performance of our ScienceBirds agent
'''
from agent.gym_hydra_agent import *
from runners.run_sb_stats import *

logging.basicConfig(format='%(name)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("sb_experiment_runner")
logger.setLevel(logging.INFO)

''' Run experiments iwth the repairing Hydra agent '''
def run_repairing_sb_experiments():
    SAMPLES = 25
    novelties = {NOVELTY: [TYPE]}
    run_performance_stats(novelties, seed=1, agent_type=AgentType.RepairingHydra, samples=SAMPLES)
    run_performance_stats(novelties, seed=1, agent_type=AgentType.Baseline, samples=SAMPLES)
    run_performance_stats(novelties, seed=1, agent_type=AgentType.Hydra, samples=SAMPLES)


if __name__ == '__main__':
    run_repairing_sb_experiments()