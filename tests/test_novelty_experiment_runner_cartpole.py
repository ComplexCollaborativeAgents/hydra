import pytest
from runners.novelty_experiment_runner_cartpole import NoveltyExperimentRunnerCartpole, NoveltyExperimentGymCartpoleDispatcher
from worlds.wsu.wsu_dispatcher import WSUObserver

#@pytest.mark.skip("Testing setting up the environment params")
def test_set_novelty():
    observer = WSUObserver()
    env_dispatcher = NoveltyExperimentGymCartpoleDispatcher(observer, render=False)
    env_dispatcher.set_novelty(novelties={'gravity': 0.9, 'masscart': 0.5, 'masspole': 0.2, 'length': 0.3, 'force_mag': 0.45})
    assert env_dispatcher.get_env_params()['gravity'] == 8.82
    assert env_dispatcher.get_env_params()['masscart'] == 0.5
    assert env_dispatcher.get_env_params()['masspole'] == 0.020000000000000004 # python computes this number
    assert env_dispatcher.get_env_params()['length'] == 0.15
    assert env_dispatcher.get_env_params()['force_mag'] == 4.5


def test_make_environment():
    observer = WSUObserver()
    env_dispatcher = NoveltyExperimentGymCartpoleDispatcher(observer, render=False)
    env_dispatcher.set_novelty(
        novelties={'gravity': 0.9, 'masscart': 0.5, 'masspole': 0.2, 'length': 0.3, 'force_mag': 0.45})
    env_dispatcher.run_trial(episode_range=range(0,0))