import pickle
import numpy as np
from sklearn import tree


class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    def __init__(self, max_number_classifier):
        '''Initialize AdaBoostClassifier

        Args:
            max_number_classifier: The maximum number of weak classifier the model can use.
        '''
        self.max_number_classifier = max_number_classifier
        self.weaker_classifier = []
        self.alpha = np.zeros((self.max_number_classifier,), dtype=np.float32)

    def is_good_enough(self):
        '''Optional'''
        pass

    def fit(self, X, y):
        '''Build a boosted classifier from the training set (X, y).

        Args:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''
        num_sample = X.shape[0]
        sample_W = np.zeros((num_sample,), dtype=np.float32)
        sample_W.fill(1.0 / num_sample)
        for i in range(self.max_number_classifier):
            clf = tree.DecisionTreeClassifier(max_depth=4)
            clf = clf.fit(X, y, sample_weight=sample_W)
            y_predict = clf.predict(X)
            error = np.sum(sample_W[np.where(y_predict != y)])
            print('Error: {}'.format(error))
            self.alpha[i] = np.log((1 - error) / error) / 2.0
            sample_W = sample_W * np.exp(-self.alpha[i] * y * y_predict) / np.sum(
                sample_W * np.exp(-self.alpha[i] * y * y_predict))
            self.weaker_classifier.append(clf)

    def predict_scores(self, X):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        nrof_sample = X.shape[0]
        scores = np.zeros((nrof_sample,), dtype=np.float32)
        for i in range(self.max_number_classifier):
            clf = self.weaker_classifier[i]
            scores += self.alpha[i] * clf.predict(X)
        return scores

    def predict(self, X, threshold=0):
        '''Predict the catagories for geven samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        return np.sign(self.predict_scores(X))

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
