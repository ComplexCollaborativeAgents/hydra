import random

import pytest
import tests.test_utils as test_utils
from runners.novelty_experiment_runner_cartpole import *
logger = test_utils.create_logger("test_cartpole")

@pytest.mark.parametrize('execution_number', range(5))
def test_agent_on_trials_without_novelties(tmp_path, execution_number):
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

    novelty_to_results = dict()

    # Select a novelty
    for experiment_counter in range(10):
        logger.info(f"---------------------------------------- iteration {experiment_counter}  ------------")
        novelties = NoveltyExperimentRunnerCartpole.generate_novelty_configs()
        for novelty in novelties:
            novelty_param, novelty_value = list(novelty['config'].items())[0]
            logger.info(f"Novelty: {novelty}")

            # Run M3
            logger.info(f"--------- M3 experiment ----------")
            observer = CartpoleHydraAgentObserver(agent_type=RepairingCartpoleHydraAgent)
            env_dispatcher = NoveltyExperimentGymCartpoleDispatcher(observer, details_directory=tmp_path)
            env_dispatcher.set_novelty(novelty['config'])
            results = env_dispatcher.run_trial(trial_number=1, episode_range=range(5))
            performance_m3 = list(results['performance'])
            for i, performance in enumerate(results['performance']):
                logger.info(f"Performance in iteration {i} is {performance}")

            # Run M4 type 1 (param name, not magnitude)
            logger.info(f"--------- M4 type 1 (param name, not magnitude) experiment ----------")
            observer = CartpoleHydraAgentObserver(agent_type=RepairingCartpoleHydraAgent)
            env_dispatcher = NoveltyExperimentGymCartpoleDispatcher(observer, details_directory=tmp_path)
            env_dispatcher.set_novelty(novelty['config'])
            env_dispatcher.set_is_known(is_known=True, novelty_info=novelty, experiment_type=1)
            results = env_dispatcher.run_trial(trial_number=1, episode_range=range(5))
            for i, performance in enumerate(results['performance']):
                logger.info(f"Performance in iteration {i} is {performance}")
            performance_m4_1 = list(results['performance'])

            # Run M4 type 2 (param name and magnitude)
            logger.info(f"--------- M4 type 2 (param name and magnitude) experiment ----------")
            observer = CartpoleHydraAgentObserver(agent_type=RepairingCartpoleHydraAgent)
            env_dispatcher = NoveltyExperimentGymCartpoleDispatcher(observer, details_directory=tmp_path)
            env_dispatcher.set_novelty(novelty['config'])
            env_dispatcher.set_is_known(is_known=True, novelty_info=novelty, experiment_type=2)
            results = env_dispatcher.run_trial(trial_number=1, episode_range=range(5))
            for i, performance in enumerate(results['performance']):
                logger.info(f"Performance in iteration {i} is {performance}")
            performance_m4_2 = list(results['performance'])

            logger.info(f"--------- Novelty {novelty} ------- ")
            for i, performance in enumerate(performance_m3):
                logger.info(f"Performance m3 in iteration {i} is {performance}")
            for i, performance in enumerate(performance_m4_1):
                logger.info(f"Performance m4_1 in iteration {i} is {performance}")
            for i, performance in enumerate(performance_m4_2):
                logger.info(f"Performance m4_2 in iteration {i} is {performance}")

            novelty_to_results[(experiment_counter, str(novelty["config"]))]= [performance_m3, performance_m4_1, performance_m4_2]

        # assert list(results['performance'])[-1]>0.8 # Assert the agent adapted to the novelty in the last iteration


    print("f----------SUMMARY--------")
    for i_novelty, results in novelty_to_results.items():
        i, novelty = i_novelty
        performance_m3, performance_m4_1, performance_m4_2 = results
        for j in range(len(performance_m3)):
            print("\t".join([str(novelty), str(i), str(performance_m3[j]), str(performance_m4_1[j]), str(performance_m4_2[j])]))

    print(novelty_to_results.items())