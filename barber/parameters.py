import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.stats import dirichlet


class PartitionParametrization:
    """A tool to convert n parameters in the range (0, 1) into a partition of
    the unit interval.

    This is slightly non-obvious - there are lots of bad ways to parametrize
    this space!

    Here we use one from numerically getting the CDF of a Dirichlet
    distribution, which if sampled uniformly in the parameters yields
    intervals uniformly distributed within the volume that has the
    correct sum.
    """
    def __init__(self, n, size=1_000_000, npoint=100):
        """Initialize the partition generator.

        Parameters
        ----------
        n: int
            The number of partition
        """
        self.n = n
        # We want something uniform so use alpha=1 for all distributions
        # The approach here is to generate a sample from all of the distributions
        # up to nbin.  We reverse this because we want to count down, e.g. if we are
        # fitting 4 bins we want to first do 4, then 3, then 2 (not 1 or zero).

        # The alphas are the parameters of the Dirichlet distribution.
        # We want them to be 1, for all our distributions.
        # If n = 4, alphas = [4, 3, 2]
        # if n = 5, alphas = [5, 4, 3, 2] etc.
        alphas =    [
            tuple([1 for i in range(nb)]) 
            for nb in range(self.n, 1, -1)
        ]
        
        # Random samples drawn from these distributions
        ds=[dirichlet.rvs(alpha, size=size) for alpha in alphas]
        
        # Generate a fitted curve to the CDF for each of these distributions,
        # by taking percentiles and fitting an interpolating curve to them.
        # Npoint is 100 by default
        q100 = np.linspace(0., 100., npoint) # uniform interval 0..100
        q1 = q100 / 100.  # uniform minterval  0..1

        # Splines of the CDF we will use
        self.cdfs = [
            # Spline from 0..1 -> CDF
            InterpolatedUnivariateSpline(q1, np.percentile(d, q100))
            for d in ds
        ]

    def __call__(self, q):
        """Convert n - 1 parameters to the n subdivions

        Parameters
        ----------
        q: array
            size n - 1, the parameters

        """
        # total partition size
        r = 1.0
        y = np.zeros(self.n)
        # for each variable, get the CDF of the flat Dirichlet
        # distribution for the remaining 
        # This corresponds to first deciding where to break the first partition
        # then the second, etc. 
        for j in range(self.n - 1):
            # partition what is left according to this split
            p_j = self.cdfs[j](q[j]) * r
            # see what is left
            r -= p_j
            y[j] = p_j
        # The last is decided just so they sum to unity.
        y[-1] = r
        return y



def partition_example():
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