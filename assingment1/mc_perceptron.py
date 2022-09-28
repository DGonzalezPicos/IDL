import numpy as np
from tqdm import tqdm


class MultiClassPerceptron(object):
    """
    Class for Multi Class Perceptrons. This is not generalized (e.g. size doesnt get determined by inputs).
    """
    def __init__(self,
                 settings=None, **kwargs):

        ### META
        self.rng = np.random.default_rng()

        ### DEFAULT VALUES
        default_settings = {
            "learning_rate": 1e-2,
            "epochs": 50,
            "weight_scale": 1e-2
        }
        if isinstance(settings, dict):
            self.settings = {**default_settings, **settings}
        else:
            self.settings = default_settings

        self.learning_rate = self.settings["learning_rate"]
        self.epochs = self.settings["epochs"]

        ### MAIN MCP
        # set initial weights, if they are initialized at w_i!=0 the first evaluation already steps them differently
        self.weights = self.rng.uniform(-1, 1, size=(257, 10)) * self.settings["weight_scale"]
        self.bias = np.zeros(shape=10)
        self.__truth = np.zeros(10)
        self.activation_func = self.__argmax

    def fit(self, X, y):
        # just the tip from the assignment sheet
        _X = self.__append_bias(X)

        for __ in tqdm(np.arange(self.epochs), leave=False):
            for x, y_train in zip(_X, y):
                # hacky solution
                self.__truth[y_train] = 1.

                # since the error is not proportional to the output (we sample categoricals/discrete)
                # we need to get the error for each node.
                # I.e. if the truth is 6, but we predict 3 this is not worse than if we predict 5.
                # So the error needs to be a vector over the outputs.
                # To change more than one node I dont use argmax but rather just the sigmoid output to get how
                # accurate the prediction was overall. E.g. we want to have low outputs for all false nodes
                # I think.
                # Maybe its better to not punish the nodes for being close even though this might lead to confusion.
                # TODO: test that out by comparing with vector of 1 at max, 0 elsewhere
                y_predicted = self.__predict(x)
                error = self.__truth - y_predicted
                update = self.learning_rate * error

                self.weights += np.outer(x, update) * self.__sigmoid_prime(y_predicted)
                self.bias += update

                self.__truth[y_train] = 0.

    def __predict(self, x):
        lin_out = np.dot(x, self.weights) + self.bias  # X @ self.weights.T + self.bias
        y_predicted = self.__sigmoid(lin_out)
        return y_predicted

    def predict(self, X):
        _X = self.__append_bias(X)
        predictions = np.zeros(len(_X))
        for i, x in enumerate(_X):
            y = self.__predict(x)
            predictions[i] = np.argmax(y)

        return predictions.flatten()

    def __argmax(self, x):
        return np.argmax(x)

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __sigmoid_prime(self, x):
        x = self.__sigmoid(x)
        return x * (1. - x)

    @staticmethod
    def __append_bias(X):
        bias = np.ones(shape=(len(X), 1))
        return np.concatenate((X, bias), axis=1)

    @staticmethod
    def accuracy(y_true, y_pred=None):
        if y_pred is not None:
            return np.mean(y_true.flatten() == y_pred.flatten())
        else:
            raise NotImplementedError


def train_mc_perceptron():
    train_in, train_out, test_in, test_out = get_data()

    percy = MultiClassPerceptron()
    percy.fit(train_in, train_out)

    predictions = percy.predict(train_in)
    print(f"Accuracy train: {percy.accuracy(train_out, predictions):.3f}")
    predictions = percy.predict(test_in)
    print(f"Accuracy test: {percy.accuracy(test_out, predictions):.3f}")


def experiment(n_samples=50):
    accuracies = np.zeros(n_samples)

    train_in, train_out, test_in, test_out = get_data()

    for i in tqdm(range(n_samples), leave=False):
        percy = MultiClassPerceptron()
        percy.fit(train_in, train_out)

        predictions = percy.predict(test_in)
        accuracies[i] = percy.accuracy(test_out, predictions)

    print(f"Accuracy (n={n_samples}): {np.median(accuracies):.3f} +/- {np.std(accuracies):.5f} (1 SD)")


def get_data():
    import pandas as pd

    df_in = pd.read_csv('train_in.csv', header=None)
    train_in = df_in.to_numpy()

    df_out = pd.read_csv('train_out.csv', header=None)
    train_out = df_out.to_numpy()

    df_in = pd.read_csv('test_in.csv', header=None)
    test_in = df_in.to_numpy()

    df_out = pd.read_csv('test_out.csv', header=None)
    test_out = df_out.to_numpy()

    return train_in, train_out, test_in, test_out


if __name__ == '__main__':
    experiment()
