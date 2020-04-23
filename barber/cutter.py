import numpy as np
import scipy.optimize as spo

def get_cuts(galaxies, nbins=3, ivals=None):
    """
    Parameters
    ----------
    galaxies: numpy.ndarray, float
        features (magnitudes and/or colors)

    Returns
    -------
    bin_assignments:
        bin assignment for each galaxy

    Notes
    -----
    sort_gals does the heavy lifting
    eval_metric is from tomo_challenge
    """

    pass


def sort_gals(galaxies, cuts):
    """
    [calls to bisect]
    """
    pass


# then call metric from tomo_challenge
