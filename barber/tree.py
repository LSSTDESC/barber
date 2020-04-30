from .base import BinClassifierMethod
from sklearn.tree import DecisionTreeClassifier

class DecisionTreeMethod(BinClassifierMethod):

    def __init__(self, *args, **kwargs):
        self.max_depth = kwargs.pop('max_depth', None)
        self.purity_test = kwargs.pop('purity_test', False)
        super().__init__(*args, **kwargs)

    def train(self, data, bins, extra_parameters):
        tree = DecisionTreeClassifier(max_depth=self.max_depth)
        tree.fit(data, bins)
        return tree

    def predict(self, tree, data, extra_parameters):

        if self.purity_test:
            p = tree.predict_proba(data)
            min_p = extra_parameters[0]
            bins = self.bins_with_threshold(p, min_p)
        else:
            bins = tree.predict(data)

        return bins
