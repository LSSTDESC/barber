import sys
sys.path.append("../tomo_challenge")
import barber.gmm
import tomo_challenge


d1 = tomo_challenge.load_data("../tomo_challenge/data/mini_training.hdf5", 
                                "riz",
                                colors=True,
                                array=True)
z1 = tomo_challenge.load_redshift("../tomo_challenge/data/mini_training.hdf5")

d2 = tomo_challenge.load_data("../tomo_challenge/data/mini_validation.hdf5",
                              "riz",
                              colors=True,
                              array=True)

z2 = tomo_challenge.load_redshift("../tomo_challenge/data/mini_validation.hdf5")



method = barber.gmm.GaussianMixtureMethod()
for nbin in range(3, 9):
    method.n_bins = nbin
    method.inform(d1, z1) # no-op
    score, _ = method.validate(d2, z2)
    print(nbin, score)
