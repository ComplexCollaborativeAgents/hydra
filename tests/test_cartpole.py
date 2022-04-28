import pytest
from agent.repair.meta_model_repair import *
from agent.cartpole_hydra_agent import CartpoleHydraAgent, CartpoleHydraAgentObserver, RepairingCartpoleHydraAgent
from agent.planning.simple_planner import *
from agent.consistency.consistency_estimator import check_obs_consistency, DEFAULT_DELTA_T
from agent.repair.cartpole_repair import CartpoleConsistencyEstimator
import gym
import tests.test_utils as test_utils
from runners.novelty_experiment_runner_cartpole import *
logger = test_utils.create_logger("test_cartpole")

@pytest.fixture(scope="module")
def launch_cartpole_sample_level():
    logger.info("Starting CartPole")
    env = gym.make("CartPole-v1")
    yield env
    # env.kill()
    logger.info("Ending CartPole")

@pytest.mark.parametrize('execution_number', range(1))
def test_agent(launch_cartpole_sample_level, execution_number):
    ''' A full system test: run the hydra cartpole agent, no non-novel cases '''

    # Setup environment and agent
    env = launch_cartpole_sample_level

    observer = CartpoleHydraAgentObserver(agent_type=RepairingCartpoleHydraAgent)
    env_dispatcher = NoveltyExperimentGymCartpoleDispatcher(observer, details_directory=".")
    results = env_dispatcher.run_trial(trial_number=1, episode_range=range(3))


