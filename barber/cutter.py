import numpy as np
import scipy.optimize as spo

TreePars = namedtuple(TreePars', ['cut_vals', 'tree_ids'])

def get_cuts(galaxies, nbins=3, tree_pars_ivals=None):
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
    (ngals, nfeat) = np.shape(galaxies)
    if ivals is None:
        cut_ivals = np.quantile(galaxies, np.linspace(0., 1., nbins), axis=1)
        assert(len(np.flatten(ivals)) == nbins**nfeat)

        # need structure and way of making dumb version of these
        tree_ids = None
        assert(len(np.unique(tree_ids)) == nbins)

        ivals = TreePars(cut_ivals, tree_ids)

    tree_pars = spo.optimize(eval_metric, ivals, args=galaxies)

    assignments = sort_gals(galaxies, tree_pars.res)

    return(assignments)


def sort_gals(galaxies, cuts):
    """
    [calls to bisect or sklearn decision tree, returns assignments]
    """
    pass


def eval_metric(galaxies, tree_pars):
    """

    """
    assignments = sort_gals(galaxies, tree_pars)
    metval = tomo_challenge.metric(assignments)
    return metval
