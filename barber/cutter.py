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
    assignments:
        bin assignment for each galaxy

    Notes
    -----
    sort_gals does the heavy lifting
    eval_metric is from tomo_challenge
    """
    if ivals is None:
        ivals = np.mean(galaxies, axis=1)
    cuts = spo.optimize(eval_metric, ivals, args=galaxies)
    assignments = sort_gals(galaxies, cuts)
    return(assignments)


def sort_gals(galaxies, cuts):
    """
    [calls to bisect or sklearn decision tree, returns assignments]
    """
    pass


def eval_metric(cuts, galaxies):
    """

    """
    assignments = sort_gals(galaxies, cuts)
    metval = tomo_challenge.metric(assignments)
    return metval
