import numpy as np
import numpy.random as npr
import scipy.optimize as spo

TreePars = namedtuple(TreePars', ['cut_vals', 'tree_ids'])

def get_cuts(galaxies, ivals=None, nbins=3):
    """
    Parameters
    ----------
    galaxies: numpy.ndarray, float
        observables (magnitudes and/or colors) to serve as features for set of galaxies
    ivals: namedtuple, numpy.ndarray, float and int
        initial values for decision tree parameters

    Returns
    -------
    assignments: numpy.ndarray, int
        bin assignment for each galaxy

    Notes
    -----
    sort_gals does the heavy lifting
    eval_metric is from tomo_challenge
    """
    (ngals, nfeat) = np.shape(galaxies)

    if tree_pars_ivals is None:
        cut_ivals = np.quantile(galaxies, np.linspace(0., 1., nbins), axis=1)
        assert(len(np.flatten(ivals)) == nbins**nfeat)

        # need structure and way of making dumb version of these
        tree_ids = npr.random_integers(0, nbins, nbins**nfeat)
        assert(len(np.unique(tree_ids)) == nbins)
        tree_ids.reshape((nfeat, nbins))

        ivals = TreePars(cut_ivals, tree_ids)

    tree_pars = spo.minimize(eval_metric, ivals, args=galaxies)

    assignments = sort_gals(galaxies, tree_pars.x)

    return(assignments)


def sort_gals(galaxies, tree):
    """
    [calls to bisect or sklearn decision tree, returns assignments]
    Parameters
    ----------

    galaxies: nfeature x n_gal array

    tree: tree object
    """
    pass


def eval_metric(galaxies, tree_pars):
    """

    """
    assignments = sort_gals(galaxies, tree_pars)
    metval = tomo_challenge.metric(assignments)
    return metval
