from .base import BinClassifierMethod
from sklearn.tree import DecisionTreeClassifier

class DecisionTreeMethod(BinClassifierMethod):

    def __init__(self, *args, **kwargs):
        self.max_depth = kwargs.pop('max_depth', None)
        self.do_rejection = kwargs.pop('do_rejection', False)
        super().__init__(*args, **kwargs)

    def train(self, data, bins, extra_parameters):
        tree = DecisionTreeClassifier(max_depth=self.max_depth)
        tree.fit(data, bins)
        return tree

    def predict(self, tree, data, extra_parameters):
        bins = tree.predict(data)

        if self.do_rejection:
            # reject objects with too low a probability here
            pass

        return bins


