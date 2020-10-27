from typing import Type

from tests.test_repairing_agent import *
from worlds.gym_cartpole_dispatcher import *
from worlds.gym_cartpole_dispatcher import GymCartpoleDispatcher
from worlds.wsu.wsu_dispatcher import WSUObserver

'''
Runner designed to running experiments for Hydra's cartpole version
'''


''' Subclass of GymCartpoleDispatcher that injects a fault '''
class FaultyGymCartpoleDispatcher(GymCartpoleDispatcher):
    def __init__(self, delegate: WSUObserver, model_id: str = 'CartPole-v1', render: bool = False):
        super().__init__(delegate, model_id, render)
        self.faults = dict() # a dictionary of env parameter to value

    def inject_fault(self, env_parameter, env_value):
        self.faults[env_parameter]=env_value

    def _make_environment(self):
        env = gym.make(self.model_id)
        for env_param in self.faults:
            env_param_value = self.faults[env_param]
            env.env.__setattr__(env_param, env_param_value)
        return env

''' SUbclass of CartpoleHydraAgent that sets the correct fluent values to the agent '''
class CartpoleOracleHydraAgentObserver(CartpoleHydraAgent):
    def __init__(self, fluent_name, fluent_value):
        super().__init__(agent_type = CartpoleHydraAgent)
        self.fluent_name = fluent_name
        self.fluent_value = fluent_value

    def trial_start(self, trial_number: int, novelty_description: dict):
        super().trial_start(trial_number, novelty_description)
        self.agent.planner.meta_model.constant_numeric_fluents[self.fluent_name]= self.fluent_value

''' Run the Hydra agent with an oracle repair, i.e., modifying the meta_model params according to the injected fault'''
def run_model_repair_experiments():
    env_param_to_fluent = dict()
    env_param_to_fluent['gravity'] = 'gravity'
    env_param_to_fluent['force_mag'] = 'force_mag'
    env_param_to_fluent['length'] = 'l_pole'
    env_param_to_fluent['masscart'] = 'm_cart'
    env_param_to_fluent['masspole'] = 'm_pole'

    env_param_to_fluent['x_threshold'] = 'x_limit'
    env_param_to_fluent['theta_threshold_radians'] = 'angle_limit'

    injected_faults = [0.8, 0.9, 1.0, 1.1, 1.2]

    for env_param in env_param_to_fluent.keys():
        fluent_name = env_param_to_fluent[env_param]
        result_file = open(path.join(settings.ROOT_PATH, "data", "cartpole", "repair", "repairing_%s_wsu.csv" % env_param), "w")
        result_file.write("Fluent\t Value\t Run\t Agent\t Iteration\t Reward\t Runtime\n")
        env_nominal_value = CartPoleMetaModel().constant_numeric_fluents[fluent_name]
        max_iterations = 5
        for fault_factor in injected_faults:
            env_param_value = env_nominal_value * fault_factor
            for i in range(max_iterations):
                observer = CartpoleHydraAgentObserver(agent_type=RepairingCartpoleHydraAgent)
                env = FaultyGymCartpoleDispatcher(observer, render=True)
                env.inject_fault(env_param, env_param_value)

                start_time = time.time()
                env.run()
                runtime = time.time() - start_time
                repair_performance = observer.agent.last_performance
                exp_name = "%s\t %s\t %s" % (env_param, fault_factor, i)
                result_file.write(
                    "%s\t %s\t %d\t %.2f\t %.2f\n" % (exp_name, "Repairing", i, repair_performance, runtime))
                result_file.flush()

                # No repair
                observer = CartpoleHydraAgentObserver(agent_type=CartpoleHydraAgent)
                env = FaultyGymCartpoleDispatcher(observer, render=True)
                env.inject_fault(env_param, env_param_value)

                start_time = time.time()
                env.run()
                runtime = time.time() - start_time
                repair_performance = observer.agent.last_performance
                exp_name = "%s\t %s\t %s" % (env_param, fault_factor, i)
                result_file.write(
                    "%s\t %s\t %d\t %.2f\t %.2f\n" % (exp_name, " No repair", i, repair_performance, runtime))
                result_file.flush()


                # Oracle
                observer = CartpoleOracleHydraAgentObserver(env_param_to_fluent[env_param], round(env_param_value,
                                                                                             CartPoleMetaModel.PLANNER_PRECISION))
                env = FaultyGymCartpoleDispatcher(observer, render=True)
                env.inject_fault(env_param, env_param_value)

                start_time = time.time()
                env.run()
                runtime = time.time() - start_time
                repair_performance = observer.agent.last_performance
                exp_name = "%s\t %s\t %s" % (env_param, fault_factor, i)
                result_file.write(
                    "%s\t %s\t %d\t %.2f\t %.2f\n" % (exp_name, "Oracle", i, repair_performance, runtime))
                result_file.flush()

        result_file.close()


if __name__ == "__main__":
    run_model_repair_experiments()