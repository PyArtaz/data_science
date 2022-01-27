import glob
import os
import time

import joblib
import numpy as np
from util_mvp import mode


class Predictor:
    def __init__(self, path=None, smoothing_samples=1, use_probability=False):
        """
        Create a Predictor object that keeps track of the past predictions and probabilities. Based on these, a
        prediction of the most likely gesture can be made.

        Parameters
        ----------
        path: str, optional
            The file path and name of the model. The default is None.
        smoothing_samples : int, optional
            The number of samples to keep track of. The default is 1.
        use_probability: bool, optional
            Whether the mode of the past predictions or their maximum probability is used for the prediction. The
            default is False.
        """
        self.model = load_model(path)
        self.pred_list = [0] * smoothing_samples
        self.prob_list = np.ones((smoothing_samples, len(self.model.classes_))) / smoothing_samples
        self.counter = 0
        self.prediction_method = 'Mode'

        if use_probability:
            self.prediction_method = 'Probability'

    def update_prediction(self, landmarks):
        """
        Update the stored prediction in pred_list and prob_list based on the passed landmarks.

        Parameters
        ----------
        landmarks: ndarray
            The landmarks of which a prediction is required.

        Returns
        -------

        """
        self.check_counter()
        # Classify landmarks using the pretrained model
        self.pred_list[self.counter - 1] = self.model.predict(landmarks)[0]
        self.prob_list[self.counter - 1, :] = self.model.predict_proba(landmarks)[0].reshape(1, -1)

    def get_prediction(self):
        """
        Get a prediction based on the prediction method. If the prediction method is 'Mode', the most common class out
        of the values stored in pred_list is taken. If the prediction method is 'Probability', the prediction is the
        class with the highest mean probability in the stored prob_list.

        Returns
        -------
        str
            The prediction based on the selected prediction method.

        """
        if self.prediction_method == 'Mode':
            prediction = str(mode(self.pred_list))
        elif self.prediction_method == 'Probability':
            probability = np.mean(self.prob_list, axis=0)
            prediction = str(self.model.classes_[np.argmax(probability)])

        return prediction

    def create_probability_dict(self):
        """
        Create and return a probability dictionary using the model classes as keys and the mean class probabilities as
        values.

        Returns
        -------
        dict
            The mean class probabilities in percent, rounded to two digits.

        """
        probability = (np.mean(self.prob_list, axis=0) * 100).round(2)
        probability_dict = dict(zip(self.model.classes_, probability))

        return probability_dict

    def get_all_predictions(self):
        """
        For development purposes only. Bypasses the selected prediction method and returns the prediction based on both,
        the mode and the highest mean probability.

        Returns
        -------
        tuple of str
            The prediction based on the mode (first) and the mean (second).

        """
        prediction_m = str(mode(self.pred_list))
        probability = np.mean(self.prob_list, axis=0)
        prediction_p = str(self.model.classes_[np.argmax(probability)])

        return prediction_m, prediction_p

    def check_counter(self):
        """
        Internal function to update the counter such that it never exceeds the number of smoothing samples.

        Returns
        -------

        """
        # Counting for rolling mode and mean calculations (Find window_width in line 90)
        if self.counter == len(self.pred_list):
            self.counter = 1
        else:
            self.counter += 1


def load_model(path=None):
    """
    Load a joblib model to predict. Default: the latest model in the path of this file, optionally any model specified
    by its file path and name.

    Parameters
    ----------
    path : str, optional
        The file path and name of the model. The default is None.

    Returns
    -------
        A joblib model

    """
    if path is None:
        directory = ''  # dataset/saved_models/
        list_of_files = glob.glob(directory + '*.sav')  # '*' means all if need specific format then e.g.: '*.sav'
        model_file = max(list_of_files, key=os.path.getctime)
    else:
        model_file = path

    # Load trained model
    model = joblib.load(model_file)

    return model


class DataLogger:
    def __init__(self, limit, threshold=0.6):
        """
        Create DataLogger object that logs the individual predictions for a time specified by limit, then computes a
        final prediction based on the logged classifications.

        Parameters
        ----------
        limit : float
            The time to log data for in seconds.
        threshold : float
            Determines with which confidence the decision for a final prediction is made. The default is 0.6.
        """
        self.time = time.time()
        self.time_limit = limit
        self.threshold = threshold
        self.log = []
        self.prediction = None

    def add_prediction(self, current_prediction):
        """
        Add a prediction to the log.

        Parameters
        ----------
        current_prediction : str
            The prediction to add to the log.

        Returns
        -------

        """
        if time.time() > self.time:
            self.log.append(current_prediction)

    def timeout(self):
        """
        Check whether the time limit is exceeded or not.

        Returns
        -------
        bool
            Whether a timeout occurred or not.

        """
        if time.time() < (self.time + self.time_limit):
            timeout = False
        else:
            timeout = True
            self.prediction = self.make_prediction()
            self.reset()
        return timeout

    def make_prediction(self):
        """
        Internal function. Takes the logs and makes a prediction based on the mode if:
            There is a sufficient amount of logged values (over 20).
            There is no more than 2 possible classifications.
            The mode of logs makes out more than threshold percentage of the logged values.

        Returns
        -------
        str
            The final prediction. Unless the above conditions are violated, then None is returned.

        """
        try:
            prediction, occ = mode(self.log, return_occurrences=True)
        except TypeError:
            return None

        elements = len(self.log)

        if elements < 20:
            prediction = None
        elif len(np.unique(self.log)) > 2:
            prediction = None
        elif occ / elements < self.threshold:
            prediction = None

        return prediction

    def final_prediction(self):
        """
        Get the final prediction.

        Returns
        -------
        str
            The final prediction. Will be None if no prediction was possible.

        """
        return self.prediction

    def reset(self):
        """
        Internal function to reset the logs and the timer.

        Returns
        -------

        """
        self.time = time.time() + 2
        self.log = []
