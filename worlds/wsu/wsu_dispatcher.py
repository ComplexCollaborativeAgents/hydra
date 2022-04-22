import copy
import queue
import random
import threading

from worlds.wsu.objects.TA2_logic import TA2Logic


class WSUObserver:

    def __init__(self):
        self.log = None
        self.possible_answers = list()

    def set_logger(self, logger):
        if self.log is None:
            self.log = logger

    def set_possible_answers(self, possible_answers):
        self.possible_answers = possible_answers

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

    def training_instance(self, feature_vector: dict, feature_label: dict) ->  \
            dict:
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
        dict
            A dictionary of your label prediction of the format {'action': label}.  This is
                strictly enforced and the incorrect format will result in an exception being thrown.
        """
        self.log.debug('Training Instance: feature_vector={}  feature_label={}'.format(
            feature_vector, feature_label))
        if feature_label not in self.possible_answers:
            self.possible_answers.append(copy.deepcopy(feature_label))

        label_prediction = random.choice(self.possible_answers)
        # novelty_probability = random.random()
        # novelty = 0
        # novelty_characterization = dict()

        return label_prediction #, novelty_probability, novelty, novelty_characterization

    def training_performance(self, performance: float, feedback: dict = None):
        """Provides the current performance on training after each instance.

        Parameters
        ----------
        performance : float
            The normalized performance score.
        feedback : dict, optional
            A dictionary that may provide additional feedback on your prediction based on the
            budget set in the TA1. If there is no feedback, the object will be None.
        """
        self.log.debug('Training Performance: {}'.format(performance))
        self.log.debug('Training Feedback: {}'.format(feedback))
        return

    def training_episode_end(self, performance: float, feedback: dict = None) -> \
            (float, float, int, dict):
        """Provides the final performance on the training episode and indicates that the training
        episode has ended.

        Parameters
        ----------
        performance : float
            The final normalized performance score of the episode.

        Returns
        -------
        float, float, int, dict
            A float of the probability of there being novelty.
            A float of the probability threshold for this to evaluate as novelty detected.
            Integer representing the predicted novelty level.
            A JSON-valid dict characterizing the novelty.
        """
        self.log.info('Training Episode End: performance={}'.format(performance))
        self.log.debug('Training Feedback: {}'.format(feedback))

        novelty_probability = random.random()
        novelty_threshold = 0.8
        novelty = 0
        novelty_characterization = dict()

        return novelty_probability, novelty_threshold, novelty, novelty_characterization

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

    def testing_start(self):
        """This is called before the trials in the novelty level/difficulty.
        """
        self.log.info('Testing Start')
        return

    def trial_start(self, trial_number: int, novelty_description: dict):
        """This is called at the start of a trial with the current 0-based number.

        Parameters
        ----------
        trial_number : int
            This is the 0-based trial number in the novelty group.
        novelty_description : dict
            A dictionary that will have a description of the trial's novelty.
        """
        self.log.info('Trial Start: #{}  novelty_desc: {}'.format(trial_number,
                                                                  str(novelty_description)))
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
            dict:
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
        dict, float, int, dict
            A dictionary of your label prediction of the format {'action': label}.  This is
                strictly enforced and the incorrect format will result in an exception being thrown.
        """
        ## commented out for evaluation
        self.log.debug('Testing Instance: feature_vector={}, novelty_indicator={}'.format(
            feature_vector, novelty_indicator))

        # Return dummy random choices, but should be determined by trained model
        if len(self.possible_answers) == 0:
            self.possible_answers = [{'action': 'right'}, {'action': 'left'}]

        label_prediction = random.choice(self.possible_answers)
        # novelty_probability = random.random()
        # novelty = random.choice(list(range(4)))
        # novelty_characterization = dict()

        return label_prediction #, novelty_probability, novelty, novelty_characterization

    def testing_performance(self, performance: float, feedback: dict = None):
        """Provides the current performance on training after each instance.

        Parameters
        ----------
        performance : float
            The normalized performance score.
        feedback : dict, optional
            A dictionary that may provide additional feedback on your prediction based on the
            budget set in the TA1. If there is no feedback, the object will be None.
        """
        return

    def testing_episode_end(self, performance: float, feedback: dict = None) -> \
            (float, float, int, dict):
        """Provides the final performance on the testing episode.

        Parameters
        ----------
        performance : float
            The final normalized performance score of the episode.
        feedback : dict, optional
            A dictionary that may provide additional feedback on your prediction based on the
            budget set in the TA1. If there is no feedback, the object will be None.

        Returns
        -------
        float, float, int, dict
            A float of the probability of there being novelty.
            A float of the probability threshold for this to evaluate as novelty detected.
            Integer representing the predicted novelty level.
            A JSON-valid dict characterizing the novelty.
        """
        self.log.info('Testing Episode End: performance={}'.format(performance))
        self.log.info('Testing Episode End: feedback={}'.format(feedback))
        
        novelty_probability = random.random()
        novelty_threshold = 0.8
        novelty = 0
        novelty_characterization = dict()

        return novelty_probability, novelty_threshold, novelty, novelty_characterization

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

    def experiment_end(self):
        """This is called when the experiment is done.
        """
        self.log.info('Experiment End')
        return


class ThreadedProcessing(threading.Thread):
    def __init__(self, callable, arguments: list, response_queue: queue.Queue):
        threading.Thread.__init__(self)
        self.callable = callable
        self.arguments = arguments
        self.response_queue = response_queue
        self.is_done = False
        return

    def run(self):
        """All work tasks should happen or be called from within this function.
        """
        result = self.callable(*self.arguments)
        self.response_queue.put(('result', result))  # adding a tuple to avoid placing a None in the queue
        return

    def stop(self):
        self.is_done = True
        return


class WSUDispatcher(TA2Logic):

    def __init__(self, delegate: WSUObserver, **kwargs):
        super().__init__(kwargs.get('config_file'),
                         kwargs.get('printout', False),
                         kwargs.get('debug', False),
                         kwargs.get('fulldebug', False),
                         kwargs.get('logfile', 'wsu-log.txt'),
                         kwargs.get('no_testing', False),
                         kwargs.get('just_one_trial', False),
                         kwargs.get('ignore_secret', False))
        self.delegate = delegate
        self.delegate.set_logger(self.log)
        self.end_training_early = True
        self.end_experiment_early = False

    def __do_processing(self, callable, arguments=None):
        if arguments is None:
            arguments = []
        response_queue = queue.Queue()
        response = None

        threaded_work = ThreadedProcessing(callable=callable,
                                           arguments=arguments,
                                           response_queue=response_queue)
        threaded_work.start()
        while response is None:
            try:
                response = response_queue.get(block=True, timeout=5)
                response = response
            except queue.Empty:
                self.log.debug("Processing AMQP Events...")
                self.process_amqp_events()

        threaded_work.stop()
        threaded_work.join()
        return response[1]

    def experiment_start(self):
        self.__do_processing(self.delegate.experiment_start)

    def training_start(self):
        self.__do_processing(self.delegate.training_start)

    def training_episode_start(self, episode_number: int):
        self.__do_processing(self.delegate.training_episode_start, [episode_number])

    def training_instance(self, feature_vector: dict, feature_label: dict) -> dict:
        return self.__do_processing(self.delegate.training_instance, [feature_vector, feature_label])

    def training_performance(self, performance: float, feedback: dict = None):
        self.__do_processing(self.delegate.training_performance, [performance, feedback])

    def training_episode_end(self, performance: float, feedback: dict = None) -> (float, float, int, dict):
        return self.__do_processing(self.delegate.training_episode_end, [performance, feedback])

    def training_end(self):
        self.__do_processing(self.delegate.training_end)

    def train_model(self):
        self.__do_processing(self.delegate.train_model)

    def save_model(self, filename: str):
        self.__do_processing(self.delegate.save_model, [filename])

    def reset_model(self, filename: str):
        self.__do_processing(self.delegate.reset_model, [filename])

    def testing_start(self):
        self.__do_processing(self.delegate.testing_start)

    def trial_start(self, trial_number: int, novelty_description: dict):
        self.__do_processing(self.delegate.trial_start, [trial_number, novelty_description])

    def testing_episode_start(self, episode_number: int):
        self.__do_processing(self.delegate.testing_episode_start, [episode_number])

    def testing_instance(self, feature_vector: dict, novelty_indicator: bool = None) -> \
            dict:
        return self.__do_processing(self.delegate.testing_instance, [feature_vector, novelty_indicator])

    def testing_performance(self, performance: float, feedback: dict = None):
        self.__do_processing(self.delegate.testing_performance, [performance, feedback])

    def testing_episode_end(self, performance: float, feedback: dict = None) -> \
            (float, float, int, dict):
        return self.__do_processing(self.delegate.testing_episode_end, [performance, feedback])

    def trial_end(self):
        self.__do_processing(self.delegate.trial_end)

    def testing_end(self):
        self.__do_processing(self.delegate.testing_end)

    def experiment_end(self):
        self.__do_processing(self.delegate.experiment_end)

