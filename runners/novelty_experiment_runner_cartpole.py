import json
import logging
logging.basicConfig(format='%(name)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("hydra_agent")

from worlds.gym_cartpole_dispatcher import GymCartpoleDispatcher
from baselines.cartpole.dqn_learner import DQNLearnerObserver, QNet
from worlds.wsu.wsu_dispatcher import WSUObserver
import os
import settings
import gym
from agent.cartpole_hydra_agent import CartpoleHydraAgent, CartpoleHydraAgentObserver, RepairingCartpoleHydraAgent, ENV_PARAM_TO_FLUENT
from agent.repair.meta_model_repair import MockMetaModelRepair
import pandas, numpy
import time
from runners import constants
import matplotlib.pyplot as plt
import seaborn as sns
import optparse


class LoggingRepairingCartpoleHydraAgent(RepairingCartpoleHydraAgent):
    def __init__(self):
        super().__init__()
        self.cnn_likelihoods = []
        self.states = []
        self.actions = []
        print("initialized logging")

    def testing_instance(self, feature_vector: dict, novelty_indicator: bool = None):
        cnn_novelties, cnn_likelihood = self.detector.detect(self.current_observation)
        self.cnn_likelihoods.append(cnn_likelihood)
        self.states.append(feature_vector)
        label = super().testing_instance(feature_vector, novelty_indicator)
        self.actions.append(label)
        return label

class OracleCartpoleHydraAgent(RepairingCartpoleHydraAgent):
    ''' Oracle agent that applies the correct repair (based on the novelty info)'''
    def __init__(self):
        super().__init__()

    def repair_meta_model(self, last_observation):
        ''' Repair the meta model based on the last observation. Uses the given novelty info data to cheat '''
        if self.novelty_info is None or len(self.novelty_info)==0:
            return super().repair_meta_model(last_observation)

        # Create perfect repair
        constant_to_repair_value = dict()
        for repair_env_param, repair_value in self.novelty_info.items():
            if repair_env_param not in ENV_PARAM_TO_FLUENT:
                self.log.info(
                    f"Repair env parameter {repair_env_param} is not mapped to any fluent in the PDDL+ model")
                continue  # Repairable constant is not a known fluent in our PDDL+
            repair_fluent_name = ENV_PARAM_TO_FLUENT[repair_env_param]

            if repair_value is None:
                return super().repair_meta_model(last_observation)

            constant_to_repair_value[repair_fluent_name]=repair_value

        new_repairable_constants = []
        new_repair_delta = []
        oracle_repair = []
        for repair_fluent_name, novel_value in constant_to_repair_value.items():
            new_repairable_constants.append(repair_fluent_name)
            new_repair_delta.append(0)
            repair_value = novel_value - self.meta_model.constant_numeric_fluents[repair_fluent_name]
            oracle_repair.append(repair_value)

        self.meta_model.repairable_constants = new_repairable_constants
        self.meta_model.repair_deltas = new_repair_delta
        mock_repair = MockMetaModelRepair(oracle_repair, self.meta_model_repair.consistency_estimator)

        # Set the repair constants and deltas in the meta_model_repair object
        repair, consistency = mock_repair.repair(self.meta_model, last_observation,
                                                       delta_t=settings.CP_DELTA_T)
        self.log.info("Repaired meta model (repair string: %s)" % repair)
        nonzero = any(map(lambda x: x != 0, repair))
        if nonzero:
            novelty_likelihood = 1.0
            self.has_repaired = True
            novelty_characterization = self.meta_model_repair.get_repair_as_json(repair)
        elif consistency > settings.CP_CONSISTENCY_THRESHOLD:
            novelty_likelihood = 1.0
            novelty_characterization = json.dumps({'Unknown novelty': 'no adjustments made'})
        else:
            novelty_likelihood = consistency / settings.CP_CONSISTENCY_THRESHOLD
            novelty_characterization = {}
        self.consistency_scores.append(consistency)
        return novelty_characterization, novelty_likelihood


class NoveltyExperimentGymCartpoleDispatcher(GymCartpoleDispatcher):
    def __init__(self, delegate: WSUObserver, model_id: str = 'CartPole-v1', render: bool = False, train_with_reward = False, log_details = False, details_directory = None):
        super().__init__(delegate, model_id, render)
        self._env_params = dict()
        self._results = pandas.DataFrame(
            columns=['episode_num', 'novelty_probability', 'novelty_threshold', 'novelty', 'novelty_characterization',
                     'performance'])
        self._is_known = None
        self._train_with_reward = train_with_reward
        self._novelty_info = None

        self._log_details = log_details
        self._details_directory = details_directory

    def get_env_params(self):
        return self._env_params

    def set_novelty(self, novelties={}):
        for param in novelties:
            self._env_params[param] = novelties[param]

    def set_is_known(self, is_known=None, novelty_info=None, experiment_type=1):
        self._is_known = is_known
        if novelty_info and 'config' in novelty_info.keys():
            self._novelty_info = {}
            for key in novelty_info['config']:
                if experiment_type == 2:
                    value = novelty_info['config'][key]
                else:
                    value = None
                self._novelty_info[key] = value

        print(self._novelty_info)


    def _make_environment(self):
        environment = gym.make(self.model_id)
        print(environment.env)
        for param in self._env_params:
            setattr(environment.env, param, self._env_params[param])
            assert getattr(environment.env, param) is self._env_params[param]
            print(param, getattr(environment.env, param))
        return environment

    def begin_experiment(self):
        self.delegate.experiment_start()

    def end_experiment(self):
        self.delegate.experiment_end()

    def run_trial(self, trial_number=0, episode_range=range(0, 1)):
        self.delegate.trial_start(trial_number=trial_number, novelty_description={})
        return (self.__run_trial(episodes=episode_range))

    def __run_trial(self, episodes: range = (0, 1), steps: int = 200):
        env = self._make_environment()
        for episode in episodes:
            observation = env.reset()
            self.delegate.testing_episode_start(episode)
            rewards = []
            features = self.observation_to_feature_vector(observation)
            reward = None
            done = False
            for step in range(1, steps + 1):
                if self.render:
                    env.render()
                    time.sleep(0.05)
                if self._train_with_reward:
                    label = self.delegate.testing_instance(feature_vector=features, novelty_indicator=self._is_known, reward=reward, done=done, novelty_info=None)
                else:
                    logger.info("Novelty info is {}".format(self._novelty_info))
                    label = self.delegate.testing_instance(feature_vector=features, novelty_indicator=self._is_known, novelty_info=self._novelty_info)
                self.log.debug("Received label={}".format(label))
                action = self.label_to_action(label)
                if done:
                    break
                observation, reward, done, _ = env.step(action)
                rewards.append(reward)

                features = self.observation_to_feature_vector(observation, 0.02 * step)

            performance = sum(rewards) / float(steps)
            # if self._log_details:
            #     self.log_details(episode)
            novelty_probability, novelty_threshold, novelty, novelty_characterization = self.delegate.testing_episode_end(performance)


            if self._results is not None:
                self._log_data(episode_num=episode, novelty_probability=novelty_probability,
                               novelty_threshold=novelty_threshold, novelty=novelty,
                               novelty_characterization=novelty_characterization, performance=sum(rewards))
        env.close()
        return self._results

    def log_details(self, episode):
        if not os.path.exists(self._details_directory):
            os.makedirs(self._details_directory)
        file = open(os.path.join(self._details_directory, "episode_{}.csv".format(str(episode))), 'w')
        log_df = pandas.DataFrame(data={
            'states': self.delegate.agent.states,
            'actions': self.delegate.agent.actions,
            'cnn_likelihood': self.delegate.agent.cnn_likelihoods
        })
        log_df.to_csv(file)
        file.close()

    def _log_data(self, episode_num=0, novelty_probability=0, novelty_threshold=0, novelty=None,
                  novelty_characterization=0, performance=0):
        result = pandas.DataFrame(data={
            'episode_num': [episode_num],
            'novelty_probability': [novelty_probability],
            'novelty_threshold': [novelty_threshold],
            'novelty': [novelty],
            'novelty_characterization': [novelty_characterization],
            'performance': [performance / constants.MAX_SCORE]
        }
        )
        self._results = self._results.append(result)


class NoveltyExperimentRunnerCartpole:
    def __init__(self, options):
        self._number_of_experiment_trials = int(options.num_trials)
        self._non_novelty_learning_trial_length = int(options.l_learning)  # TODO: Remove this, does not seem to be used.
        self._non_novelty_performance_trial_length = int(options.l_performance)
        self._novelty_trial_length = int(options.l_novelty)
        self._agent_type = options.agent
        self._experiment_type = options.experiment_type
        self._novelty_config = options.novelty_config

        self._results_directory_path = os.path.join(settings.ROOT_PATH, "runners", "experiments", "cartpole", options.name, options.agent)
        if not os.path.exists(self._results_directory_path):
            os.makedirs(self._results_directory_path)


        self._results_details_directory_path = None
        self._log_episode_details = False
        if options.log_episode_details == 'True':
            self._results_details_directory_path = os.path.join(self._results_directory_path, "details")
            self._log_episode_details = True


    def run_experiment_subtrial(self, episode_range, trial_num, trial_type, episode_type, novelty_id, novelty):
        observer = None
        if self._agent_type == 'dqn':
            observer = DQNLearnerObserver()
        if self._agent_type == 'basic':
            observer = CartpoleHydraAgentObserver(agent_type=CartpoleHydraAgent)
        if self._agent_type == 'oracle':
            observer = CartpoleHydraAgentObserver(agent_type=OracleCartpoleHydraAgent)
        if self._agent_type == 'repairing':
            if self._log_episode_details == True:
                print("logging")
                observer = CartpoleHydraAgentObserver(agent_type=LoggingRepairingCartpoleHydraAgent)
            else:
                observer = CartpoleHydraAgentObserver(agent_type=RepairingCartpoleHydraAgent)

        assert observer is not None
        logger.info("Running agent type {}".format(self._agent_type))
        env_dispatcher = NoveltyExperimentGymCartpoleDispatcher(observer, render=False, train_with_reward=(self._agent_type == 'dqn'), log_details = self._log_episode_details, details_directory=os.path.join(self._results_details_directory_path, trial_type, str(trial_num)))
        print("experiment type {}".format(self._experiment_type))
        if trial_type == constants.UNKNOWN:
            env_dispatcher.set_is_known(None, None, self._experiment_type)
        else:
            if episode_type == constants.NOVELTY:
                print("episode type {}".format(constants.NOVELTY))
                if self._experiment_type == 2 or self._experiment_type == 3:
                    print("detected {}".format(True))
                    env_dispatcher.set_is_known(True, novelty, self._experiment_type)
                else:
                    env_dispatcher.set_is_known(True, None, self._experiment_type)
            else:
                if episode_type == constants.NON_NOVELTY_PERFORMANCE:
                    env_dispatcher.set_is_known(False, None, self._experiment_type)

        if episode_type == constants.NOVELTY:
            env_dispatcher.set_novelty(novelty['config'])

        results = env_dispatcher.run_trial(trial_number=trial_num, episode_range=episode_range)
        results['trial_num'] = trial_num
        results['novelty_id'] = novelty_id
        results['trial_type'] = trial_type
        results['episode_type'] = episode_type
        results['level'] = novelty['level']
        results['env_config'] = str(novelty['config'])

        return results

    def run_experiment(self, novelty_config=None):
        if novelty_config is None:
            novelties_config = NoveltyExperimentRunnerCartpole.generate_novelty_configs()
        else:
            novelties_config = [novelty_config]
        for novelty in novelties_config:
            results_file_handle = open(os.path.join(self._results_directory_path, "novelty_{}_{}.csv".format(novelty['uid'], self._experiment_type)), "a")
            results_dataframe = pandas.DataFrame(columns=['episode_num','novelty_probability','novelty_threshold','novelty','novelty_characterization','performance','trial_num','novelty_id','trial_type','episode_type','level','env_config'])
            results_dataframe.to_csv(results_file_handle, index=False)
            #for trial_type in [constants.UNKNOWN, constants.KNOWN]:
            for trial_type in [constants.KNOWN]:
                for trial in range(0, self._number_of_experiment_trials):
                    episode_num = 0
                    subtrial_result = self.run_experiment_subtrial(
                        episode_range=range(episode_num, episode_num + self._non_novelty_performance_trial_length),
                        trial_num=trial,
                        trial_type=trial_type,
                        episode_type=constants.NON_NOVELTY_PERFORMANCE,
                        novelty_id=novelty['uid'],
                        novelty=novelty)
                    subtrial_result.to_csv(results_file_handle, index=False, header=False)
                    episode_num = episode_num + self._non_novelty_performance_trial_length
                    subtrial_result = self.run_experiment_subtrial(
                        episode_range=range(episode_num, episode_num + self._novelty_trial_length),
                        trial_num=trial,
                        trial_type=trial_type,
                        episode_type=constants.NOVELTY,
                        novelty_id=novelty['uid'],
                        novelty=novelty)
                    subtrial_result.to_csv(results_file_handle, index=False, header=False)


    @staticmethod
    def generate_novelty_configs():
        novelty_config = {
            'levels': {
                1: [constants.MASSCART, constants.LENGTH, constants.FORCE_MAG],
                2: [constants.GRAVITY]
            },
            'values': range(5, 25, 5)
        }

        novelties = []
        uid = 0
        for level in novelty_config['levels'].keys():
            for param in novelty_config['levels'][level]:
                for value in novelty_config['values']:
                    novelty = {
                        'uid': "{}_{}_{}".format(param, value, uid),
                        'level': level,
                        'config': {param: constants.ENV_PARAMS_NOMINAL[param] * value / 10}
                    }
                    novelties.append(novelty)
                    uid = uid + 1
        return novelties

    @staticmethod
    def categorize_examples_for_novelty_detection(dataframe):
        dataframe['is_novel'] = numpy.where(dataframe['novelty_probability'] < dataframe['novelty_threshold'], False, True)
        dataframe['TN'] = numpy.where((dataframe['episode_type'] == constants.NON_NOVELTY_PERFORMANCE) & (dataframe['is_novel'] == False), 1, 0)
        dataframe['FP'] = numpy.where((dataframe['episode_type'] == constants.NON_NOVELTY_PERFORMANCE) & (dataframe['is_novel'] == True), 1, 0)
        dataframe['TP'] = numpy.where((dataframe['episode_type'] == constants.NOVELTY) & (dataframe['is_novel'] == True),1, 0)
        dataframe['FN'] = numpy.where((dataframe['episode_type'] == constants.NOVELTY) & (dataframe['is_novel'] == False),1, 0)
        return dataframe

    @staticmethod
    def get_trials_summary(dataframe):
        trials = dataframe[['trial_num', 'trial_type', 'novelty_id', 'env_config', 'FN', 'FP', 'TN', 'TP', 'performance']].groupby(
            ['trial_type','novelty_id', 'trial_num']).agg({'FN': numpy.sum, 'FP': numpy.sum, 'TN': numpy.sum, 'TP': numpy.sum, 'performance': numpy.mean})
        trials['is_CDT'] = numpy.where((trials['TP'] > 1) & (trials['FP'] == 0), True, False)
        cdt = trials[trials['is_CDT'] == True]
        return trials, cdt

    @staticmethod
    def get_program_metrics(cdt: pandas.DataFrame, trials: pandas.DataFrame):
        num_trials_per_type = trials.groupby("trial_type").agg({'FN': len}).rename(columns={'FN': 'count'})
        scores = cdt.groupby("trial_type").agg({'FN': numpy.mean, 'FP': len}).rename(columns={'FN': 'M1', 'FP': 'M2'})
        scores['M2'] = scores['M2'] / num_trials_per_type['count']
        scores['NRP'] = trials.groupby("trial_type").agg({'performance': numpy.mean})
        return scores

    @staticmethod
    def plot_experiment_results(df, novelty_episode_number):
        plt.figure(figsize=(16, 9))
        ax = sns.lineplot(data=df, y='performance', x='episode_num', hue='trial_type', ci=95)
        ax.set(ylim=(0, 1.1))
        plt.axvline(x=novelty_episode_number, color='red')
        plt.title("Experiment results", fontsize=20)
        plt.xlabel("episodes", fontsize=15)
        plt.ylabel("performance", fontsize=15)

if __name__ == '__main__':
    parser = optparse.OptionParser(usage="usage: %prog [options]")
    # parser.add_option("--agent",
    #                   dest='agent',
    #                   help='name of the agent you want to run: basic, repairing, dqn',
    #                   default='dqn')
    # parser.add_option("--name",
    #                   dest="name",
    #                   help="name of the directory in which all the results will be stored at ../data/cartpole/",
    #                   default="jun7")
    # parser.add_option("--num_trials",
    #                   dest='num_trials',
    #                   help="Number of full trials to be run. Each trial is several subtrials",
    #                   default=5)
    # parser.add_option("--learning_subtrial",
    #                   dest='l_learning',
    #                   help='number of episodes in the learning subtrial',
    #                   default=0)
    # parser.add_option("--performance-subtrial",
    #                   dest='l_performance',
    #                   help='number of episodes in the non-novelty performance subtrial',
    #                   default=5)
    # parser.add_option("--novelty-subtrial",
    #                   dest='l_novelty',
    #                   help='number of episodes in the novelty subtrial',
    #                   default=50)
    # parser.add_option("--log-episode-details",
    #                   dest='log_episode_details',
    #                   help='if we want to record states, action, cnn_likelihood, consistency_scores',
    #                   default=False)
    # parser.add_option("--novelty_config",
    #                   dest='novelty_config',
    #                   help='a dict of novelty configurations',
    #                   default={'uid': 'length_1point1_masscart_point9', 'level': 1, 'config': {constants.LENGTH: 1.1, constants.MASSCART: 0.9}})

    parser.add_option("--agent",
                      dest='agent',
                      help='name of the agent you want to run: basic, repairing, dqn, oracle',
                      default='repairing')
    parser.add_option("--name",
                      dest="name",
                      help="name of the directory in which all the results will be stored at ../data/cartpole/",
                      default="m3m4")
    parser.add_option("--num_trials",
                      dest='num_trials',
                      help="Number of full trials to be run. Each trial is several subtrials",
                      default=1)
    parser.add_option("--learning_subtrial",
                      dest='l_learning',
                      help='number of episodes in the learning subtrial',
                      default=0)
    parser.add_option("--performance-subtrial",
                      dest='l_performance',
                      help='number of episodes in the non-novelty performance subtrial',
                      default=1)
    parser.add_option("--novelty-subtrial",
                      dest='l_novelty',
                      help='number of episodes in the novelty subtrial',
                      default=30)
    parser.add_option("--log-episode-details",
                      dest='log_episode_details',
                      help='if we want to record states, action, cnn_likelihood, consistency_scores',
                      default='True')
    parser.add_option("--novelty_config",
                      dest='novelty_config',
                      help='a dict of novelty configurations',
                      default={'uid': 'length_1point1_gravity_12', 'level': 1, 'config': {constants.LENGTH: 1.1, constants.GRAVITY: 12}})
    parser.add_option("--experiment_type",
                     dest='experiment_type',
                     help='a numeral representing if to run 1: no info; 2: with full info, 3: with only fluent names',
                     default=2)


    (options, args) = parser.parse_args()
    print(options)


    experiment_runner = NoveltyExperimentRunnerCartpole(options)
    if options.novelty_config:
        experiment_runner.run_experiment(options.novelty_config)
    else:
        experiment_runner.run_experiment()


