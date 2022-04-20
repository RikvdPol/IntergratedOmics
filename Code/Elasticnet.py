from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import ElasticNet


class Elasticnet:
    def __init__(self, file, labelname):
        self.file = file
        self.labelname = labelname
        self.labels = None

    def extract_labels(self):
        self.labels = self.file[self.labelname]
        print(self.labels)

    def split_data(self):
        pass

    def define_model(self):
        model = ElasticNet(alpha=1.0, l1_ratio=0.5)
        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

    def train_model(self):
        pass

    def predict(self):
        pass

    def evaluate_model(self):
        pass

