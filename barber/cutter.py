import numpy as np
import numpy.random as npr
import scipy.optimize as spo
import tomo_challenge.metrics as tcm

# custom data type, could be replaced with/tie in to tree.py class
# cut_vals is (nfeat, nbins - 1) numpy array, float
# tree_ids is ((nbins,) * nfeat) numpy array, int
TreePars = namedtuple('TreePars', ['cut_vals', 'tree_ids'])

# should maybe put this function in a class so we can call TreePars.to_array
def treepars_to_array(treepars):
    """
    Flattens cut_vals and tree_ids for optimizer
    """
    cuts = np.flatten(treepars.cut_vals)
    ids = np.flatten(treepars.tree_ids)
    arr = np.concatenate((cuts, ids))
    return(arr)

# should maybe put this function in a class so we can call TreePars.from_array
def array_to_treepars(arr):
    """
    Converts optimizer format of 1D array back into namedtuple of arrays
    """
    flat_cuts = arr[type(arr) == float]
    flat_ids = arr[type(arr) == int]
    nbins = len(np.unique(flat_ids))
    nfeat = len(flat_cuts) / (nbins - 1)
    # maybe do some assert checks with these just in case types have problems
    # cuts = arr[0:nfeat*(nbins-1)].reshape((nfeat, nbins-1))
    # ids = arr[feat*(nbins-1):].reshape((nbins,) * nfeat)
    cuts = flat_cuts.reshape((nfeat, nbins-1))
    ids = flat_ids.reshape((nbins,) * nfeat)
    treepars = TreePars(cuts, ids)
    return(treepars)

def get_cuts(galaxies, ival_treepars=None, nbins=3):
    """
    Obtains simplest possible bin definitions: cuts in the space of observables given number of bins

    Parameters
    ----------
    galaxies: numpy.ndarray, float
        observables (magnitudes and/or colors and/or errors) to serve as features for set of galaxies
        shape(galaxies) = (ngals, nfeat)
    ival_treepars: namedtuple, numpy.ndarray, float and int, optional
        initial values for decision tree parameters
        shape(ivals.cut_vals) = (nfeat, (nbins - 1))
        shape(tree_ids) = ((nbins,) * nfeat)
    nbins: int, optional
        number of bins for which to obtain cuts

    Returns
    -------
    assignments: numpy.ndarray, int
        bin assignment for each galaxy
        shape(assignments) = (ngals, 1)

    Notes
    -----
    `sort_gals` does the heavy lifting.
    `eval_metric` will call one of the metrics from [tomo_challenge](https://github.com/LSSTDESC/tomo_challenge/blob/master/tomo_challenge/metrics.py).
    The original idea for a general, non-cut-based optimizer was to have parameters equal to the (ngals) length array of ints representing the bin assignments, but that's not necessary for the simple cut-and-sweep barber and would probably break `spo.minimize`.
    """
    (ngals, nfeat) = np.shape(galaxies)

    if ival_treepars is None:
        cut_ivals = np.quantile(galaxies, np.linspace(0., 1., nbins), axis=1)
        assert(len(np.flatten(ivals)) == nbins**nfeat)

        # need structure and way of making dumb version of these
        tree_ids = npr.random_integers(0, nbins, nbins**nfeat)
        assert(len(np.unique(tree_ids)) == nbins)
        tree_ids.reshape((nfeat, nbins))

        ival_treepars = TreePars(cut_ivals, tree_ids)

    ivals = treepars_to_array(ival_treepars)
    opt_res = spo.minimize(eval_metric, ivals, args=galaxies)
    treepars = array_to_treepars(opt_res.x)

    assignments = sort_gals(galaxies, treepars)

    return(assignments)

def sort_gals(galaxies, tree_pars):
    """
    Divides available galaxies into subsets according to a given decision tree on their observables

    Parameters
    ----------
    galaxies: nfeature x n_gal array

    tree: tree object

    Notes
    -----
    could be based on bisect, or maybe a sklearn object?
    """
    pass


def eval_metric(arr, galaxies):
    """
    Just calls a metric from tomo_challenge wrapped for the `spo.minimize` API

    Notes
    -----
    Replace `tcm.metric` with actual call to one of the tomo_challenge metrics
    Actually, there's a problem in that the current tomo_challenge metrics require the true redshifts...
    """
    treepars = array_to_treepars(arr)
    assignments = sort_gals(galaxies, treepars)
    metval = tcm.metric(assignments)
    return metval
