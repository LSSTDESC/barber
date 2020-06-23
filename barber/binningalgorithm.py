# Superclass structures establishing a unified API for algorithms that assign galaxies to bins

import numpy as np
import tomo_challenge as tc
from tc.metrics import compute_snr_score

class BinningAlgorithm(object):
    """
    A superclass for any algorithm that defines and executes a tomographic binning scheme
    """
    def __init__(self, quiet=False, *args, **kwargs):
        """
        Parameters
        ----------
        quiet: boolean, optional
            suppresses progress updates to stdout

        Notes
        -----
        Perhaps there should also be a generic ingestion method to unpack the `hyperparams` dictionary into subclass attributes.
        """
        self.hyperparams = kwargs
        self.n_bins = None
        self.metric = None

    def assess(self, bin_assignments, metric=(compute_snr_score, **args)):
        """
        Evaluates a metric or objective function to optimize, currently restricted to the tomo challenge metric

        Parameters
        ----------
        bin_assignments: numpy.ndarray, int
            integer bin choice for each object being assessed
        metric: tuple, (tomo_challenge.Metric object, **args), optional
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
        For an unsupervised classifier or optimizer, this method would wrap the `fit()` function and possibly define `n_bins`.
        Either way, the contents of this method will likely access `self.hyperparams` defined by `self.inform()`.
        """
        raise NotImplementedError("Implement the `assign()` method in subclasses")
        # return(bin_assignments)

    def inform(self, training_data, training_target, **kwargs):
        """
        Employs any information used to train or define a prior

        Parameters
        ----------
        training_data: numpy.ndarray, float
            (n_galaxies, n_features) array of training data
        training_target: numpy.ndarray, int
            (n_galaxies) array of target values of training set

        Returns
        -------
        self.assign(): method
            the function that assigns bin numbers to galaxies

        Notes
        -----
        For a supervised classifier, this method would wrap the training function that calls the `self.assess()` method and would define `self.n_bins` from the number of unique values in training_target.
        The `**kwargs` may include tuning parameters of the training process.
        The hyperparameters of the trained model would be rolled into `self.hyperparams` that would then be accessed by the `self.assign()` method.
        """
        raise NotImplementedError("Implement the `inform()` method in subclasses")
        # self.hyperparams =
        # return(self.assign())

    def validate(self, validation_data, validation_target):
        """
        A shortcut to `assign()` and then `assess()` on data with known truth

        Parameters
        ----------
        validation_data: numpy.ndarray, float
            (n_galaxies, n_features) array of validation data
        validation_target: numpy.ndarray, int
            (n_galaxies) array of redshift values of validation set

        Returns
        -------
        score: float
            metric value
        bin_assignments: numpy.ndarray, int
            (n_galaxies) array of bin assignments

        Notes
        -----
        The interaction between the input parameter of `training_data` and the way the `self.assess()` method is called here is almost certainly broken for any metric besides `compute_snr_metric`.
        To work generically, it should include a check that `self.metric` has already been defined and whether it requires `validation_target` or any other arguments.
        Also, the returned parameters might be better off as a `NamedTuple`.
        """
        bin_assignments = self.assign(validation_data, n_bins=self.n_bins)
        score = self.assess(validation_data, (self.metric, validation_target))
        return((score, bin_assignments))

class SupervisedBinner(BinningAlgorithm):
    """
    Temporary wrapper for `barber.base.BinClassifierMethod` to demonstrate correspondence
    """
    def __init__(self, nbin,
                 training_data, training_z,
                 validation_data, validation_z, quiet=False):
        """
        Parameters
        ----------
        quiet: boolean, optional
            suppresses progress updates to stdout

        Notes
        -----
        """
        self.hyperparams = {'nbin': nbin,
                            'training_data': training_data,
                            'training_z': training_z,
                            'validation_data': validation_data,
                            'validation_z': validation_z}
        self.n_bins = None
        self.metric = BinClassifierMethod(nbin,
                     training_data, training_z,
                     validation_data, validation_z, quiet=False)).metric()

    def assess(self, bin_assignments):
        """
        Evaluates a metric or objective function to optimize, currently restricted to the tomo challenge metric

        Parameters
        ----------
        bin_assignments: numpy.ndarray, int
            integer bin choice for each object being assessed
        metric: tuple, (tomo_challenge.Metric object, **args), optional
            a metric provided by the `tomo_challenge` and any parameters it requires beyond the bin assignments, such as true values of the redshifts

        Returns
        -------
        score: float
            the value of the metric or objective function

        Notes
        -----
        """
        score = self.metric(bin_assignments)
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
        For an unsupervised classifier or optimizer, this method would wrap the `fit()` function and possibly define `n_bins`.
        Either way, the contents of this method will likely access `self.hyperparams` defined by `self.inform()`.
        """
        raise NotImplementedError("Implement the `assign()` method in subclasses")
        # return(bin_assignments)

    def inform(self, training_data=None, training_target=None, extra_starts=None, **kwargs):
        """
        Employs any information used to train or define a prior

        Parameters
        ----------
        training_data: numpy.ndarray, float
            (n_galaxies, n_features) array of training data
        training_target: numpy.ndarray, int
            (n_galaxies) array of target values of training set

        Returns
        -------
        self.assign(): method
            the function that assigns bin numbers to galaxies

        Notes
        -----
        For a supervised classifier, this method would wrap the training function that calls the `self.assess()` method and would define `self.n_bins` from the number of unique values in training_target.
        The `**kwargs` may include tuning parameters of the training process.
        The hyperparameters of the trained model would be rolled into `self.hyperparams` that would then be accessed by the `self.assign()` method.
        """
        training_data = self.hyperparams['training_data']
        training_target = self.hyperparams['training_z']
        self.hyperparams['']
        raise NotImplementedError("Implement the `inform()` method in subclasses")
        # self.hyperparams =
        # return(self.assign())

    def validate(self, validation_data, validation_target):
        """
        A shortcut to `assign()` and then `assess()` on data with known truth

        Parameters
        ----------
        validation_data: numpy.ndarray, float
            (n_galaxies, n_features) array of validation data
        validation_target: numpy.ndarray, int
            (n_galaxies) array of redshift values of validation set

        Returns
        -------
        score: float
            metric value
        bin_assignments: numpy.ndarray, int
            (n_galaxies) array of bin assignments

        Notes
        -----
        The interaction between the input parameter of `training_data` and the way the `self.assess()` method is called here is almost certainly broken for any metric besides `compute_snr_metric`.
        To work generically, it should include a check that `self.metric` has already been defined and whether it requires `validation_target` or any other arguments.
        Also, the returned parameters might be better off as a `NamedTuple`.
        """
        bin_assignments = self.assign(validation_data, n_bins=self.n_bins)
        score = self.assess(validation_data, (self.metric, validation_target))
        return((score, bin_assignments))
