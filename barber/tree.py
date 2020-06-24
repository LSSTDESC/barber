from .binningalgorithm import BinningAlgorithm
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from .parameters import PartitionParametrization
from scipy.optimize import minimize
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
                 max_depth=None, purity_test=False,
                 quiet=False):
        """
        Parameters
        ----------
        quiet: boolean, optional
            suppresses progress updates to stdout

        Notes
        -----
        """
        super().__init__(n_bins, max_depth=max_depth, purity_test=purity_test, quiet=quiet)
        # could move to superclass
        self.partitioner = PartitionParametrization(n_bins)
        self.last_model = None


    def inform(self, training_data, training_z):
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
        # optimize the conversion of 
        # fit 

        # parameters describing bin edges
        start = [0.0 for i in range(self.n_bins - 1)]

        if self.hyperparams['purity_test']:
            start.append(0.5)

        print("Optimizing")

        res = minimize(
                lambda p: -self._evaluate_params(p, training_data, training_z),
                start,
                method='powell',
        )
        score = -res.fun
        print("\nDone.\n")

        self.informed = self._make_and_predict(res.x, training_data, training_z)[0]

        return self.informed


    def _make_and_predict(self, p, training_data, training_z):
        """
        Turn parameters into a model
        """
        if self.hyperparams['purity_test']:
            bin_edges = self._unit_parameters_to_z_edges(p[:-1], training_z)
            min_p = p[-1]
        else:
            bin_edges = self._unit_parameters_to_z_edges(p, training_z)
            min_p = -np.inf

        if not self.quiet:
            print(f"Params: {p}  min_p: {min_p:.2f}  edges: {bin_edges}", end="")


        # Assign each galaxy in the training data to a bin
        training_bin = np.digitize(training_z, bin_edges, right=True) - 1

        # Build a model (e.g. tree) to fit this data set and
        # get its score
        model, bin_prediction = self._build_and_predict(training_data, training_bin, min_p)
        return model, bin_prediction        

    def _evaluate_params(self, p, training_data, training_z):
        """
        Turn parameters into a score
        """
        model, bin_prediction = self._make_and_predict(p, training_data, training_z)
        # Get the score - customizable metric
        score = self.assess(bin_prediction, training_z)

        if not self.quiet:
            print(f"    Score: {score:.2f}", end="\r")

        return score


    def _build_and_predict(self, training_data, training_bin, min_p):
        """
        Turn bins into a model and prediction of that model
        """
        tree = DecisionTreeClassifier(max_depth=self.hyperparams['max_depth'])
        tree.fit(training_data, training_bin)
        model = (tree, min_p)
        bin_prediction = self.assign(training_data, model)
        return model, bin_prediction


    def assign(self, testing_data, model=None):
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
        if model is None:
            model = self.informed

        tree, min_p = model

        if self.hyperparams['purity_test']:
            p = tree.predict_proba(testing_data)
            bin_assignments = self._bins_with_threshold(p, min_p)
        else:
            bin_assignments = tree.predict(testing_data)

        return bin_assignments




    def _unit_parameters_to_z_edges(self, p, z):
        zmin = z.min()
        zrange = z.max() - z.min()
        n1 = len(p)
        n2 = self.n_bins - 1
        if not n1 == n2:
            raise ValueError("Wrong size (should be {n2} but is {n1})")
        # Convert to a partition of the unit interval
        q = self.partitioner(p)
        # And convert that to a partition of zmin .. zmax.  Add a small delta to each side so
        # that all objects are included
        z_edges = np.concatenate([[zmin-1e-6], zmin + zrange * np.cumsum(q)])
        z_edges[-1] += 1e-6
        return z_edges


    # use later
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