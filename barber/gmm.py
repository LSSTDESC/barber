from .base import UnsupervisedClusteringMethod


class GaussianMixtureMethod:
    def cluster(self, data):
        model = mixture.GaussianMixture(n_components=self.nbin,
                                        covariance_type='full')
        Y = model.fit_predict(data)

        return Y
