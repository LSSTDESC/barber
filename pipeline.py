import sys
sys.path.append("../tomo_challenge")
import tomo_challenge

from barber import *
# is there a way to import all subclasses of BinningAlgorithm without using metaclasses?
# ooh, maybe steal from tomo_challenge.base.Tomographer?

training_data = tomo_challenge.load_data("../tomo_challenge/data/mini_training.hdf5",
                                "riz",
                                colors=True,
                                array=True)
training_target = tomo_challenge.load_redshift("../tomo_challenge/data/mini_training.hdf5")

validation_data = tomo_challenge.load_data("../tomo_challenge/data/mini_validation.hdf5",
                              "riz",
                              colors=True,
                              array=True)

validation_target = tomo_challenge.load_redshift("../tomo_challenge/data/mini_validation.hdf5")

for tomographer in [gmm.GaussianMixtureModel(), tree.DecisionTree()]:

    for nbin in range(3, 9):
        tomographer.n_bins = nbin
        tomographer.inform(training_data, training_target) # no-op
        score, _ = tomographer.validate(validation_data, validation_target)
        print(tomographer, nbin, score)

if __name__ == '__main__':
    # Command line arguments
    try:
        bands = sys.argv[1]
        n_bin_max = int(sys.argv[2])
        assert bands in ['riz', 'griz']
    except:
        sys.stderr.write("Script takes two arguments, 'riz'/'griz' and n_bin_max\n")
        sys.exit(1)

    # Run main code
    for n_bin in range(1, n_bin_max+1):
        scores = main(bands, n_bin)
