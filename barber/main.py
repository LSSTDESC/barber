import sys
sys.path.append("../tomo_challenge")
from . import gmm
from . import tree
import tomo_challenge

def main():
    d1 = tomo_challenge.load_data("../tomo_challenge/data/training.hdf5", 
                                    "riz",
                                    colors=True,
                                    array=True)
    z1 = tomo_challenge.load_redshift("../tomo_challenge/data/training.hdf5")

    d2 = tomo_challenge.load_data("../tomo_challenge/data/validation.hdf5",
                                  "riz",
                                  colors=True,
                                  array=True)

    z2 = tomo_challenge.load_redshift("../tomo_challenge/data/validation.hdf5")



    for nbin in range(3, 9):
        method = gmm.GaussianMixtureMethod(n_bins = nbin)
        method.inform(d1, z1) # no-op
        score, _ = method.validate(d2, z2)
        print('GMM', nbin, score)


    for nbin in range(3, 9):
        method = tree.DecisionTree(n_bins = nbin)
        method.inform(d1, z1)
        score, _ = method.validate(d2, z2)
        print('Tree\n', nbin, score)


if __name__ == '__main__':
    main()
