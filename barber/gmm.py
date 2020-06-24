from .binningalgorithm import BinningAlgorithm
from sklearn import mixture

class GaussianMixtureMethod(BinningAlgorithm):

    def inform(self, training_data, training_target, **kwargs):
        # as an unsupervised method, inform does nothing
        pass

    def assign(self, test_data):
        model = mixture.GaussianMixture(n_components=self.n_bins,
                                        covariance_type='full')
        Y = model.fit_predict(test_data)

        return Y
