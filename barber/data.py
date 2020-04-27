import os
import h5py
import numpy as np


def load(data_set, thin=1):
    # path on nersc
    if 'NERSC_HOST' in os.environ:
        data_path = "/global/cfs/cdirs/lsst/www/txpipe/tomo_challenge_data/ugrizy/"
    else:
        data_path = './data/'
    # Load training data
    f = h5py.File(f"{data_path}/{data_set}.hdf5", "r")

    # mags - will also try a version with the colours
    r = f['r_mag'][:]
    i = f['i_mag'][:]
    z = f['z_mag'][:]
    training_data = np.array([r, i, z]).T.clip(10, 30)
    del r, i, z

    # Thin
    training_data = training_data[::thin]

    # training redshifts
    redshift = f['redshift_true'][:][::thin]

    f.close()
    return training_data, redshift

