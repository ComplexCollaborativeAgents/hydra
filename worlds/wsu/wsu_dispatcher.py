import copy
import random
from worlds.wsu.objects.TA2_logic import TA2Logic


class WSUObserver:

    def __init__(self):
        self.log = None
        self.possible_answers = list()

    def set_logger(self, logger):
        if self.log is None:
            self.log = logger

    def experiment_start(self):
        """This function is called when this TA2 has connected to a TA1 and is ready to begin
        the experiment.
        """
        self.log.info('Experiment Start')
        return

    def training_start(self):
        """This function is called when we are about to begin training on episodes of data in
        your chosen domain.
        """
        self.log.info('Training Start')
        return

    def training_episode_start(self, episode_number: int):
        """This function is called at the start of each training episode, with the current episode
        number (0-based) that you are about to begin.

        Parameters
        ----------
        episode_number : int
            This identifies the 0-based episode number you are about to begin training on.
        """
        self.log.info('Training Episode Start: #{}'.format(episode_number))
        return

    def training_instance(self, feature_vector: dict, feature_label: dict) -> (dict, bool, int):
        """Process a training

        Parameters
        ----------
        feature_vector : dict
            The dictionary of the feature vector.  Domain specific feature vector formats are
            defined on the github (https://github.com/holderlb/WSU-SAILON-NG).
        feature_label : dict
            The dictionary of the label for this feature vector.  Domain specific feature labels
            are defined on the github (https://github.com/holderlb/WSU-SAILON-NG). This will always
            be in the format of {'action': label}.  Some domains that do not need an 'oracle' label
            on training data will receive a valid action chosen at random.

        Returns
        -------
        dict, bool, int
            A dictionary of your label prediction of the format {'action': label}.  This is
                strictly enforced and the incorrect format will result in an exception being thrown.
            A boolean as to whether agent detects novelty.
            Integer representing the predicted novelty level.
        """
        self.log.debug('Training Instance: feature_vector={}  feature_label={}'.format(
            feature_vector, feature_label))
        if feature_label not in self.possible_answers:
            self.possible_answers.append(copy.deepcopy(feature_label))

        label_prediction = random.choice(self.possible_answers)
        novelty_detected = False
        novelty = 0

        return label_prediction, novelty_detected, novelty

    def training_performance(self, performance: float):
        """Provides the current performance on training after each instance.

        Parameters
        ----------
        performance : float
            The normalized performance score.
        """
        self.log.debug('Training Performance: {}'.format(performance))
        return

    def training_episode_end(self, performance: float):
        """Provides the final performance on the training episode and indicates that the training
        episode has ended.

        Parameters
        ----------
        performance : float
            The final normalized performance score of the episode.
        """
        self.log.info('Training Episode End: performance={}'.format(performance))
        return

    def training_end(self):
        """This function is called when we have completed the training episodes.
        """
        self.log.info('Training End')
        return

    def train_model(self):
        """Train your model here if needed.  If you don't need to train, just leave the function
        empty.  After this completes, the logic calls save_model() and reset_model() as needed
        throughout the rest of the experiment.
        """
        self.log.info('Train the model here if needed.')
        return

    def save_model(self, filename: str):
        """Saves the current model in memory to disk so it may be loaded back to memory again.

        Parameters
        ----------
        filename : str
            The filename to save the model to.
        """
        self.log.info('Save model to disk.')
        return

    def reset_model(self, filename: str):
        """Loads the model from disk to memory.

        Parameters
        ----------
        filename : str
            The filename where the model was stored.
        """
        self.log.info('Load model from disk.')
        return

    def novelty_start(self):
        """This indicates the start of a series of trials at a novelty level/difficulty.
        """
        self.log.info('Novelty Space Start')
        return

    def testing_start(self):
        """This is called before the trials in the novelty level/difficulty.
        """
        self.log.info('Testing Start')
        return

    def trial_start(self, trial_number: int):
        """This is called at the start of a trial with the current 0-based number.

        Parameters
        ----------
        trial_number : int
            This is the 0-based trial number in the novelty group.
        """
        self.log.info('Trial Start: #{}'.format(trial_number))
        return

    def testing_episode_start(self, episode_number: int):
        """This is called at the start of each testing episode in a trial, you are provided the
        0-based episode number.

        Parameters
        ----------
        episode_number : int
            This is the 0-based episode number in the current trial.
        """
        self.log.info('Testing Episode Start: #{}'.format(episode_number))
        return

    def testing_instance(self, feature_vector: dict, novelty_indicator: bool = None) -> \
            (dict, bool, int):
        """Evaluate a testing instance.  Returns the predicted label or action, if you believe
        this episode is novel, and what novelty level you beleive it to be.

        Parameters
        ----------
        feature_vector : dict
            The dictionary containing the feature vector.  Domain specific feature vectors are
            defined on the github (https://github.com/holderlb/WSU-SAILON-NG).
        novelty_indicator : bool, optional
            An indicator about the "big red button".
                - True == novelty has been introduced.
                - False == novelty has not been introduced.
                - None == no information about novelty is being provided.

        Returns
        -------
        dict, bool, int
            A dictionary of your label prediction of the format {'action': label}.  This is
                strictly enforced and the incorrect format will result in an exception being thrown.
            A boolean as to whether agent detects novelty.
            Integer representing the predicted novelty level.
        """
        self.log.debug('Testing Instance: feature_vector={}, novelty_indicator={}'.format(
            feature_vector, novelty_indicator))

        # Return dummy random choices, but should be determined by trained model
        label_prediction = random.choice(self.possible_answers)
        novelty_detected = random.choice([True, False])
        novelty = random.choice(list(range(4)))

        return label_prediction, novelty_detected, novelty

    def testing_performance(self, performance: float):
        """Provides the current performance on training after each instance.

        Parameters
        ----------
        performance : float
            The normalized performance score.
        """
        return

    def testing_episode_end(self, performance: float):
        """Provides the final performance on the testing episode.

        Parameters
        ----------
        performance : float
            The final normalized performance score of the episode.
        """
        self.log.info('Testing Episode End: performance={}'.format(performance))
        return

    def trial_end(self):
        """This is called at the end of each trial.
        """
        self.log.info('Trial End')
        return

    def testing_end(self):
        """This is called after the trials in a novelty level/difficulty are completed.
        """
        self.log.info('Testing End')
        return

    def novelty_end(self):
        """This is called when we are done with a novelty level/difficulty.
        """
        self.log.info('Novelty Space End')
        return

    def experiment_end(self):
        """This is called when the experiment is done.
        """
        self.log.info('Experiment End')
        return


class WSUDispatcher(TA2Logic):

    def __init__(self, delegate: WSUObserver, **kwargs):
        super().__init__(kwargs.get('config_file'),
                         kwargs.get('printout', False),
                         kwargs.get('debug', False),
                         kwargs.get('fulldebug', False),
                         kwargs.get('logfile', 'wsu-log.txt'))
        self.delegate = delegate
        self.delegate.set_logger(self.log)

    def experiment_start(self):
        self.delegate.experiment_start()

    def training_start(self):
        self.delegate.training_start()

    def training_episode_start(self, episode_number: int):
        self.delegate.training_episode_start(episode_number)

    def training_instance(self, feature_vector: dict, feature_label: dict) -> (dict, bool, int):
        return self.delegate.training_instance(feature_vector, feature_label)

    def training_performance(self, performance: float):
        self.delegate.training_performance(performance)

    def training_episode_end(self, performance: float):
        self.delegate.training_episode_end(performance)

    def training_end(self):
        self.delegate.training_end()

    def train_model(self):
        self.delegate.train_model()

    def save_model(self, filename: str):
        self.delegate.save_model(filename)

    def reset_model(self, filename: str):
        self.delegate.reset_model(filename)

    def novelty_start(self):
        self.delegate.novelty_start()

    def testing_start(self):
        self.delegate.testing_start()

    def trial_start(self, trial_number: int):
        self.delegate.trial_start(trial_number)

    def testing_episode_start(self, episode_number: int):
        self.delegate.testing_episode_start(episode_number)

    def testing_instance(self, feature_vector: dict, novelty_indicator: bool = None) -> \
            (dict, bool, int):
        return self.delegate.testing_instance(feature_vector, novelty_indicator)

    def testing_performance(self, performance: float):
        self.delegate.testing_performance(performance)

    def testing_episode_end(self, performance: float):
        self.delegate.testing_episode_end(performance)

    def trial_end(self):
        self.delegate.trial_end()

    def testing_end(self):
        self.delegate.experiment_end()

    def novelty_end(self):
        self.delegate.novelty_end()

    def experiment_end(self):
        self.delegate.experiment_end()

