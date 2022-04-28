import random

import pytest
import tests.test_utils as test_utils
from runners.novelty_experiment_runner_cartpole import *
logger = test_utils.create_logger("test_cartpole")

@pytest.mark.parametrize('execution_number', range(5))
def test_agent(tmp_path, execution_number):
    ''' A full system test: run the hydra cartpole agent, no non-novel cases '''

    observer = CartpoleHydraAgentObserver(agent_type=RepairingCartpoleHydraAgent)
    env_dispatcher = NoveltyExperimentGymCartpoleDispatcher(observer,
                                                            details_directory=tmp_path)
    results = env_dispatcher.run_trial(trial_number=1, episode_range=range(3))

    for i in results['performance']:
        assert i>0.95

@pytest.mark.parametrize('execution_number', range(1))
def test_agent_on_trials_with_novelties(tmp_path, execution_number):
    ''' A full system test: run the hydra cartpole agent, no non-novel cases '''

    observer = CartpoleHydraAgentObserver(agent_type=RepairingCartpoleHydraAgent)
    env_dispatcher = NoveltyExperimentGymCartpoleDispatcher(observer,
                                                            details_directory=tmp_path)

    # Add a novelty
    novelties = NoveltyExperimentRunnerCartpole.generate_novelty_configs()
    novelty = random.choice(novelties)
    logger.info(f"Novelty: {novelty}")
    env_dispatcher.set_novelty(novelty['config'])

    # Run the agent
    results = env_dispatcher.run_trial(trial_number=1, episode_range=range(5))

    for i, performance in enumerate(results['performance']):
        logger.info(f"Performance in iteration {i} is {performance}")

    assert list(results['performance'])[-1]>0.8 # Assert the agent adapted to the novelty in the last iteration

