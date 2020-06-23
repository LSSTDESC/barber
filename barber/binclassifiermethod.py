from .parameters import PartitionParametrization
from tomo_challenge.metrics import compute_snr_score
from scipy.optimize import minimize
import numpy as np


# This object evaluates a proposed splitting into a pre-defined number of bins.
# It first maps nbin-1 parameters to the partition using a Splitter
# It makes a Decision Tree to fit to that splitting, and then applies it to training data,
# and then runs the challenge metric
class BinClassifierMethod:
    def __init__(self, nbin,
                 training_data, training_z,
                 validation_data, validation_z, quiet=False):
        self.nbin = nbin
        self.quiet = quiet
        # Data sets
        self.training_data = training_data
        self.training_z = training_z
        self.validation_data = validation_data
        self.validation_z = validation_z

        self.partitioner = PartitionParametrization(nbin)
        self.zmin = training_z.min()
        self.zmax = training_z.max()
        self.zrange = self.zmax - self.zmin


    def __call__(self, p):
        # Split into partition parameters and any extra parameters
        r = p[self.nbin - 1:]
        p = p[:self.nbin - 1]

        z_edges = self.unit_parameters_to_z_edges(p)
        score = self.evaluate_edges(z_edges, r)
        return score

    def unit_parameters_to_z_edges(self, p):
        n1 = len(p)
        n2 = self.nbin - 1
        if not n1 == n2:
            raise ValueError("Wrong size (should be {n2} but is {n1})")
        # Convert to a partition of the unit interval
        q = self.partitioner(p)
        # And convert that to a partition of zmin .. zmax.  Add a small delta to each side so
        # that all objects are included
        z_edges = np.concatenate([[self.zmin-1e-6], self.zmin + self.zrange * np.cumsum(q)])
        z_edges[-1] += 1e-6
        return z_edges

    def evaluate_edges(self, bin_edges, extra_parameters):
        if not self.quiet:
            print(f"Edges: {bin_edges}   Extra: {extra_parameters}")

        # Assign each galaxy in the training data to a bin
        training_bin = np.digitize(self.training_z, bin_edges, right=True) - 1

        # Subclasses should provide methods to train on one data set and
        # then predict on another.  Unsupervised methods might just return
        # None for the first method and do everything in the second.
        # Methods can ignore rejection criterion if they wish.
        model = self.train(self.training_data, training_bin, extra_parameters)
        bin_prediction = self.predict(model, self.validation_data, extra_parameters)

        # Get the score - customizable metric
        score = self.metric(bin_prediction)

        if not self.quiet:
            print(f"Score: {score}\n")
        return score

    def metric(self, bin_prediction):
        """Compute a metric for fitting.

        Currently this uses the tomo challenge metric.

        Parameters
        ----------
        bin_prediction: array
            integer bin choice for each object in the validation
            sample. Can be -1 for no choice

        Returns
        -------
        score: float
        """
        counts = np.bincount(bin_prediction)
        for i, c in enumerate(counts):
            print(f"{c} objects in bin {i}")

        return compute_snr_score(bin_prediction, self.validation_z)

    def optimize(self, extra_starts=None, **kwargs):
        """Run scipy.minimize on self to fit edges.

        Parameters
        ----------

        **kwargs: dict
            Any parameters to be passed to minimize
        """
        start = [0.5 for i in range(self.nbin - 1)]

        if extra_starts is not None:
            start = np.concatenate([start, extra_starts])

        res = minimize(
            lambda p: -self(p), start, **kwargs)

        return self.unit_parameters_to_z_edges(res.x[:self.nbin - 1]), res.x[self.nbin-1:], -res.fun

    def bins_with_threshold(self, probability, min_probability):
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
        bins[reject] = self.nbin
        return bins


    def train(self, training_bin, extra_parameters):
        raise NotImplementedError("Implement the train method in subclasses")

    def predict(self, model, training_bin):
        raise NotImplementedError("Implement the predict method in subclasses")


class UnsupervisedClusteringMethod:
    def __init__(self, nbin,
                 validation_data, validation_z, quiet=False):
        self.nbin = nbin
        self.validation_data = validation_data
        self.validation_z = validation_z

    def run(self):
        bins = self.cluster(self.validation_data)
        return self.metric(bin_prediction)

    def metric(self, bin_prediction):
        """Compute a metric for fitting.

        Currently this uses the tomo challenge metric.

        Parameters
        ----------
        bin_prediction: array
            integer bin choice for each object in the validation
            sample. Can be -1 for no choice

        Returns
        -------
        score: float
        """
        return compute_snr_score(bin_prediction, self.validation_z)

    def cluster(self, data):
        raise NotImplementedError("Implement the  method in subclasses")
