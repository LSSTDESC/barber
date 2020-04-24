import h5py
import os
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.stats import dirichlet
from scipy.optimize import minimize
from sklearn.tree import DecisionTreeClassifier

# my nersc path
import sys
sys.path.append("/global/cscratch1/sd/zuntz/tomo_challenge")
import tomo_challenge.metrics


def load():
    # path on nersc
    if 'NERSC_HOST' in os.environ:
        data_path = "/global/cfs/cdirs/lsst/www/txpipe/tomo_challenge_data/ugrizy/"
    else:
        data_path = './data/'

    thin = 100

    # Load training data
    f = h5py.File(data_path + "training.hdf5", "r")

    # mags - will also try a version with the colours
    r = f['r_mag'][:]
    i = f['i_mag'][:]
    z = f['z_mag'][:]
    training_data = np.array([r, i, z]).T.clip(10, 30)
    training_data = training_data[::thin]
    del r, i, z

    # training redshifts
    redshift = f['redshift_true'][:][::thin]

    f.close()
    return training_data, redshift


# This converts n parameters in the range (0, 1) to (n+1) partitions in the range (0, 1)
# This is slightly non-obvious - there are lots of bad ways to parametrize this space
# Here we use one from numerically getting the CDF of a Dirichlet distribution, which if
# sampled uniformly in the parameters yields points uniform in the 
class Splitter:
    def __init__(self, nbin, size=1_000_000):
        self.nbin = nbin
        # We want something uniform so use alpha=1 for all distributions
        # The approach here is to generate a sample from all of the distributions
        # up to nbin.  We reverse this because we want to count down, e.g. if we are
        # fitting 4 bins we want to first do 4, then 3, then 2 (not 1 or zero).
        alphas = [tuple([1 for i in range(nb)]) for nb in range(1,self.nbin+1)][1:][::-1]
        
        # random values drawn from these distributions
        ds=[dirichlet.rvs(alpha, size=size) for alpha in alphas]
        
        # Generate a fitted curve to the CDF for each of these distributions,
        # by taking percentiles and fitting an interpolating curve to them
        q100 = np.linspace(0., 100., 100) # uniform interval 0..100
        q1 = q100 / 100.  # uniform minterval  0..1
        self.cdfs = [
            # Spline from 0..1 -> CDF
            InterpolatedUnivariateSpline(q1, 
                                         np.percentile(d, q100)) # CDF
            for d in ds
        ]

    def __call__(self, q):
        # total partition size
        r = 1.0
        y = np.zeros(self.nbin)
        # for each variable, get the CDF of the flat Dirichlet
        # distribution for the remaining 
        # This corresponds to first deciding where to break the first partition
        # then the second, etc. 
        for j in range(self.nbin - 1):
            # partition what is left according to this split
            p_j = self.cdfs[j](q[j]) * r
            # see what is left
            r -= p_j
            y[j] = p_j
        # The last is decided just so they sum to unity.
        y[-1] = r
        return y


def splitter_example():
    import matplotlib.pyplot as plt
    s = Splitter(3)
    d = np.zeros((1000,3))
    for i in range(1000):
        x = np.random.uniform(0, 1, 3)
        y = s(x)
        d[i] = y
    plt.subplot(1,2,1)
    plt.scatter(d[:,0], d[:,1], s=1)
    plt.subplot(1,2,2)
    plt.scatter(d[:,0], d[:,2], s=1)
    plt.show()


# This object evaluates a proposed splitting into a pre-defined number of bins.
# It first maps nbin-1 parameters to the partition using a Splitter
# It makes a Decision Tree to fit to that splitting, and then applies it to training data,
# and then runs the challenge metric
class BinEvaluator:
    def __init__(self, data, z, nbin, metric=tomo_challenge.metrics.compute_snr_score):
        self.nbin = nbin
        self.data = data
        self.z = z
        self.splitter = Splitter(nbin)
        self.zmin = z.min()
        self.zmax = z.max()
        self.zrange = self.zmax - self.zmin
        self.metric = metric

    def __call__(self, p):
        z_edges = self.unit_parameters_to_z_edges(p)
        score = self.evaluate_edges(z_edges)
        z_str = ', '.join([f'{zi:.3f}' for zi in z_edges])
        print(f'params {p}: z_edges: {z_str}  score: {score:.2f}')
        return score
    
    def unit_parameters_to_z_edges(self, p):
        if not len(p) == self.nbin - 1:
            raise ValueError("Wrong size")
        # Convert to a partition of the unit interval
        q = self.splitter(p)
        # And convert that to a partition of zmin .. zmax
        z_edges = np.concatenate([[self.zmin], self.zmin + self.zrange * np.cumsum(q)])
        return z_edges

    def evaluate_edges(self, bin_edges):
        # Make a single decision tree (could also try a random forest,
        # though this will be much much slower). Will bepossible though.
        tree = DecisionTreeClassifier()
        cut = (self.z >= bin_edges[0]) & (self.z < bin_edges[-1])

        # Make -1 (unused) the default bin
        bin_sel = np.repeat(-1, self.z.size)

        # Fill in the bins for all used galaxies
        bin_sel[cut] = np.digitize(self.z[cut], bin_edges, right=True) - 1

        # Train the tree to replicate these
        tree.fit(self.data, bin_sel)

        # Predict - we should change to using a second validation set for this
        bin_guess = tree.predict(self.data)

        # Get the score - customizable metric
        score = self.metric(bin_guess, self.z)
        return score

def main():
    training_data, redshift = load()
    B = BinEvaluator(training_data, redshift, 5)

    # Need to be careful with choice of optimizer.  The default one doesn't work - it takes tiny delta steps, which
    # don't change the bin that any galaxies are in, so the metric comes out as flat and it declares success right
    # at the start.  Nelder-Mead takes bigger steps.  I spent a year working in the basement at UCL trying to get
    # numerical optimizers to fit 6-parameter galaxy models for shape fitting.  All optimizers are bad.
    res = minimize(lambda p: -B(p), [0.5, 0.5, 0.5, 0.5], method='Nelder-Mead', tol=0.05)

    print(B.unit_parameters_to_z_edges(res.x))
