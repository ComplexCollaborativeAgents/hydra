from agent.repairing_hydra_agent import RepairingHydraSBAgent
from agent.hydra_agent import HydraAgent
import pytest
from agent.repair.meta_model_repair import *
from agent.planning.sb_meta_model import *
import worlds.science_birds as sb
import pickle
import tests.test_utils as test_utils
import os.path as path
from agent.repair.sb_repair import ScienceBirdsConsistencyEstimator, ScienceBirdsMetaModelRepair

GRAVITY_FACTOR = "gravity_factor"
BASE_LIFE_WOOD_MULTIPLIER = "base_life_wood_multiplier"
BASE_MASS_WOOD_MULTIPLIER = "base_mass_wood_multiplier"
DATA_DIR = path.join(settings.ROOT_PATH, 'data')
TEST_DATA_DIR = path.join(DATA_DIR, 'science_birds', 'tests')

# Setup environment and agent
save_obs = True
plot_exp_vs_obs = False
settings.SB_SIM_SPEED = 5
settings.SB_GT_FREQ = 1

fh = logging.FileHandler("hydra_debug.log",mode='w')
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger = test_utils.create_logger("hydra_agent")
logger.setLevel(logging.INFO)
logger.addHandler(fh)

'''
A greedy best-first search model repair implementation. 
'''
class MockMetaModelRepair(SimulationBasedMetaModelRepair):
    def __init__(self, oracle_repair):
        meta_model = ScienceBirdsMetaModel()

        super().__init__(meta_model.repairable_constants,
                         ScienceBirdsConsistencyEstimator(),
            [1.0] * len(meta_model.repairable_constants))

        self.oracle_repair = oracle_repair

    ''' Repair the given domain and plan such that the given plan's expected outcome matches the observed outcome'''
    def repair(self,
               pddl_meta_model: ScienceBirdsMetaModel,
               observation, delta_t=1.0):

        self.current_delta_t = delta_t
        self.current_meta_model = pddl_meta_model

        repair = [0] * len(self.oracle_repair)  # Repair is a list, in order of the fluents_to_repair list
        base_consistency = self._compute_consistency(repair, observation)
        best_consistency = self._compute_consistency(self.oracle_repair, observation)

        assert(best_consistency<base_consistency)

        self._do_change(self.oracle_repair)

        return self.oracle_repair, best_consistency



@pytest.fixture(scope="module")
def launch_science_birds_with_all_levels():
    logger.info("Starting ScienceBirds")
    env = sb.ScienceBirds(None,launch=True,config='test_repair_wood_health.xml')
    yield env
    env.kill()
    logger.info("Ending ScienceBirds")

''' Run an experiment'''
def _run_experiment(hydra, experiment_name, max_iterations = 10):
    # Inject fault and run the agenth
    # _inject_fault_to_meta_model(hydra.meta_model, GRAVITY_FACTOR)
    try:
        results_file = open(path.join(TEST_DATA_DIR, "%s.csv" % experiment_name), "w")
        results_file.write("Iteration\t Reward\t PlanningTime\t CummulativePlanningTime\t base_life_wood_multiplier\t base_mass_wood_multiplier\n")

        iteration = 0
        obs_with_rewards = 0
        cummulative_reward = 0
        while iteration < max_iterations:
            hydra.run_next_action()
            observation = hydra.find_last_obs()

            results_file.write("%d\t %.2f\t %.2f\t %.2f\t %.4f\t %.4f\n" % (iteration,
                                                                     observation.reward,
                                                                     hydra.overall_plan_time,
                                                                     hydra.cumulative_plan_time,
                                                                     hydra.meta_model.constant_numeric_fluents[BASE_LIFE_WOOD_MULTIPLIER],
                                                                     hydra.meta_model.constant_numeric_fluents[BASE_MASS_WOOD_MULTIPLIER]))
            results_file.flush()

            # Store observation for debug
            if save_obs:
                obs_file = path.join(TEST_DATA_DIR, "%s-%d.obs" % (experiment_name, iteration))  # For debug
                meta_model_file = path.join(TEST_DATA_DIR, "%s-%d.mm" % (experiment_name, iteration))  # For debug
                pickle.dump(observation, open(obs_file, "wb"))  # For debug
                pickle.dump(hydra.meta_model, open(meta_model_file, "wb"))
            if plot_exp_vs_obs:
                test_utils.plot_expected_vs_observed(hydra.meta_model, observation)

            if observation.reward > 0:
                logger.info("Reward ! (%.2f), iteration %d" % (observation.reward, iteration))
                obs_with_rewards = obs_with_rewards + 1
                cummulative_reward += observation.reward
            iteration = iteration + 1

        return len(hydra.completed_levels), cummulative_reward
    finally:
        results_file.close()

@pytest.mark.skip("Have not migrated to 0.3.6 yet")
def test_set_of_levels_no_repair(launch_science_birds_with_all_levels):
    env = launch_science_birds_with_all_levels
    hydra = HydraAgent(env)
    max_iterations = 10
    _run_experiment(hydra, "no_repair-%d" % max_iterations, max_iterations=max_iterations)

@pytest.mark.skip("No test found: ./Levels/novelty_level_2/type8/Levels/test_alt2_1_1_8_2.xml")
def test_set_of_levels_repair_no_fault():
    max_iterations = 10

    logger.info("Starting no repair experiment")
    env = sb.ScienceBirds(None,launch=True,config='test_repair_wood_health.xml')
    hydra = HydraAgent(env)
    levels_completed_no_repair, reward_no_repair = _run_experiment(hydra, "with_repair-%d" % max_iterations, max_iterations=max_iterations)
    env.kill()
    logger.info("Ending no repair experiment, levels completed = %d, reward = %.2f" % (levels_completed_no_repair, reward_no_repair))

    logger.info("Starting mock oracle repair experiment")
    env = sb.ScienceBirds(None,launch=True,config='test_repair_wood_health.xml')
    hydra = RepairingHydraSBAgent(env)
    hydra.meta_model_repair = MockMetaModelRepair([3.0, 3.0])
    levels_completed_oracle_repair, reward_with_oracle_repair = _run_experiment(hydra, "with_repair-%d" % max_iterations, max_iterations=max_iterations)
    env.kill()
    logger.info("Ending mock oracle repair experiment, levels completed = %d, reward = %.2f" % (levels_completed_oracle_repair, reward_with_oracle_repair))
    assert(levels_completed_no_repair<=levels_completed_oracle_repair)

    logger.info("Starting repair experiment")
    env = sb.ScienceBirds(None,launch=True,config='test_repair_wood_health.xml')
    hydra = RepairingHydraSBAgent(env)
    levels_completed_repair, reward_with_repair = _run_experiment(hydra, "with_repair-%d" % max_iterations, max_iterations=max_iterations)
    env.kill()
    logger.info("Ending repair experiment, levels completed = %d, reward = %.2f" % (levels_completed_repair, reward_with_repair))
    assert(levels_completed_no_repair<=levels_completed_repair)




''' Inject a fault to the agent's meta model '''
def _inject_fault_to_meta_model(meta_model : ScienceBirdsMetaModel, fluent_to_change = GRAVITY_FACTOR):
    meta_model.constant_numeric_fluents[fluent_to_change] = 6.0

@pytest.mark.skip("Have not migrated to 0.3.6 yet")
def test_set_of_levels_repair_with_fault(launch_science_birds_with_all_levels):
    env = launch_science_birds_with_all_levels
    hydra = RepairingHydraSBAgent(env)
    _inject_fault_to_meta_model(hydra.meta_model, GRAVITY_FACTOR)
    max_iterations = 10
    _run_experiment(hydra, "with_repair_bad_gravity-%d" % max_iterations, max_iterations=max_iterations)

@pytest.mark.skip("Have not migrated to 0.3.6 yet")
def test_set_of_levels_no_repair_with_fault(launch_science_birds_with_all_levels):
    env = launch_science_birds_with_all_levels
    hydra = HydraAgent(env)
    _inject_fault_to_meta_model(hydra.meta_model, GRAVITY_FACTOR)
    max_iterations = 10
    _run_experiment(hydra, "no_repair_bad_gravity-%d" % max_iterations, max_iterations=max_iterations)



''' Tests the faster implementation of the simulator '''
@pytest.mark.skip("Profiling test.")
def test_debugging():
    obs_output_file = path.join(settings.ROOT_PATH, "data", "science_birds", "tests",
                                "test_repair.p")  # For debug
    obs = pickle.load(open(obs_output_file, "rb"))

    mm_output_file = path.join(settings.ROOT_PATH, "data", "science_birds", "tests",
                               "test_repair.mm")  # For debug
    meta_model = pickle.load(open(mm_output_file, "rb"))


    import cProfile

    profiler = cProfile.Profile()
    profiler.enable()
    repair_algorithm = ScienceBirdsMetaModelRepair()
    best_repair, best_consistency = repair_algorithm.repair(meta_model, obs, settings.SB_DELTA_T)
    profiler.disable()
    profiler.print_stats()
    profiler.dump_stats("repair.profile")