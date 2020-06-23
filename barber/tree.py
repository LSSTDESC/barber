from .base import BinClassifierMethod
from sklearn.tree import DecisionTreeClassifier
#
# # class DecisionTreeMethod(BinClassifierMethod):
#
#     # def __init__(self, *args, **kwargs):
#     #     self.max_depth = kwargs.pop('max_depth', None)
#     #     self.purity_test = kwargs.pop('purity_test', False)
#     #     super().__init__(*args, **kwargs)
#
#     # def train(self, data, bins, extra_parameters):
#     #     tree = DecisionTreeClassifier(max_depth=self.max_depth)
#     #     tree.fit(data, bins)
#     #     return tree
#
#     def predict(self, tree, data, extra_parameters):
#
#         if self.purity_test:
#             p = tree.predict_proba(data)
#             min_p = extra_parameters[0]
#             bins = self.bins_with_threshold(p, min_p)
#         else:
#             bins = tree.predict(data)
#
#         return bins

class DecisionTree(BinningAlgorithm):
    """
    DecisionTree classifier as subclass of BinningAlgorithm
    """
    def __init__(self, n_bins,
                 max_depth=False, purity_test=False,
                 quiet=False):
        """
        Parameters
        ----------
        quiet: boolean, optional
            suppresses progress updates to stdout

        Notes
        -----
        """
        super().__init__(*args, **kwargs)
        self.n_bins = n_bins

    def assign(self, testing_data, min_p=None):
        """
        Assigns bins to galaxies

        Parameters
        ----------
        testing_data: numpy.ndarray, float
            (n_galaxies, n_features) array of test data
        min_p: float, optional
            probability threshold

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
        tree = self.informed

        if self.purity_test:
            p = tree.predict_proba(testing_data)
            bin_assignments = self._bins_with_threshold(p, min_p)
        else:
            bin_assignments = tree.predict(testing_data)

        return bin_assignments

    def inform(self, training_data, training_target):
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
        data = np.hstack(training_data, training_target)
        tree = DecisionTreeClassifier(max_depth=self.max_depth)
        tree.fit(data, self.n_bins)
        self.informed = tree
        return tree

    def _bins_with_threshold(self, probability, min_probability):
        """
        Turn a probabillity per bin per object into a choice of bins,
        where objects whose highest probability is still below the
        threshold are assigned a new bin, above all the rest.

        This could be a utility with nbin as a third input.

        Parameters
        ----------
        probability: array n_gal x n_bin
            estimated probability of each galaxy being in each bin
        min_probability: float
            A threshold value below which galaxies are deemed too
            ambiguous to put in their primary bin and are relagated to a new one.

        """
        bins = np.argmax(probability, axis=1)
        # There must be a fancy indexing way of doing this
        best_p = np.array([probability[i,b] for i,b in enumerate(bins)])
        reject = best_p < min_probability
        n_reject = reject.sum()
        reject_fraction = n_reject / reject.size
        print(f"{n_reject} objects ({reject_fraction:.2%}) fail purity test")
        bins[reject] = self.n_bins
        return bins
