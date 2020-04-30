# Any algorithm for assigning galaxies to "bins" should share a unified API defined by this superclass

import numpy as np
import tomo_challenge as tc
from tc.metrics import compute_snr_score

class BinningAlgorithm:
    def __init__(self, quiet=False):
        """
        """
        self.vb = quiet
        self.n_bins = None
        self.metric = None

    def assess(self, bin_assignments, metric=(compute_snr_score, **args)):
        """
        Evaluates a metric or objective function to optimize

        Currently this uses the tomo challenge metric.

        Parameters
        ----------
        bin_assignments: numpy.ndarray, int
            integer bin choice for each object being assessed
        metric: tuple, (tomo_challenge.Metric object, **kwargs), optional
            a metric provided by the `tomo_challenge` and any parameters it requires beyond the bin assignments, such as true values of the redshifts

        Returns
        -------
        score: float
            the value of the metric or objective function

        Notes
        -----
        The `challenge_metric` input assumes `compute_snr_score` will soon be joined by more metrics with a unified API defined by a superclass.
        `barber` may include additional subclassed metrics.
        I suspect the syntax of `**args` appearing in the `metric` input parameters is not kosher and, regardless, should probably be replaced with a dictionary.
        """
        if self.metric is None:
            self.metric = metric[0]
        score = self.metric(bin_assignments, **args)
        return(score)

    def assign(self, test_data, n_bins=None):
        """
        Assigns bins to galaxies

        Parameters
        ----------
        test_data: numpy.ndarray, float
            (n_galaxies, n_features) array of test data
        n_bins: int, optional
            the number of bins, if it is not informed by test data nor fit by the algorithm itself

        Returns
        -------
        bin_assignments: numpy.ndarray, int
            (n_galaxies) array of target values of test set

        Notes
        -----
        For a supervised classifier, this method would wrap the `predict()` function.
        For an unsupervised classifier, this method would wrap the `fit()` function and possibly define `n_bins`.
        """
        pass

    def inform(self, training_data, training_target, **kwargs):
        """
        Ingests any data used to train or define a prior

        Parameters
        ----------
        training_data: numpy.ndarray, float
            (n_galaxies, n_features) array of training data
        training_target: numpy.ndarray, int
            (n_galaxies) array of target values of training set

        Notes
        -----
        For a supervised classifier, this method would wrap the training function that calls the `self.assess()` method and would define `self.n_bins` from the number of unique values in training_target.
        """
        pass

    def validate(self, validation_data, validation_target):
        """
        Parameters
        ----------
        training_data: numpy.ndarray, float
            (n_galaxies, n_features) array of training data
        training_target: numpy.ndarray, int
            (n_galaxies) array of target values of training set

        Returns
        -------
        score:
        bin_assignments:

        Notes
        -----
        The interaction between the input parameter of `training_data` and the way the `self.assess()` method is called here is almost certainly broken for any metric besides `compute_snr_metric`.
        To work generically, it should include a check that `self.metric` has already been defined and whether it requires `validation_target` or any other arguments.
        Also, the returned parameters might be better off as a `NamedTuple`.
        """
        bin_assignments = self.assign(validation_data, n_bins=self.n_bins)
        score = self.assess(validation_data, (self.metric, validation_target))
        return((score, bin_assignments))
