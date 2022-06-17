from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


# paste this code under the other class i suppose
class RF(AbstractAlgorithm):
    
    def define_model(self, n_splits=10, n_repeats=3, random_state=None):
        cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

        model = RandomForestClassifier()
        
        return model, cv

if __name__ == '__main__':
    # testing
    data = pd.read_csv('Covariates.csv', sep='\t')
    data = data.dropna(axis=0)
    X = data[['Age', 'BMI']].values
    y = data['Gender'].values

    model= RandomForestClassifier()

    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(scores)
