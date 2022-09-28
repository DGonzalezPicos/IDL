import numpy as np
from tqdm import tqdm

class MultiClassPerceptron(object):
    def __init__(self,
                 settings=None, **kwargs):

        ### META
        self.rng = np.random.default_rng()

        ### DEFAULT VALUES
        default_settings = {
            "learning_rate": 1e-2,
            "epochs": 100,
            "weight_scale": 1e-2
        }
        if isinstance(settings, dict):
            self.settings = {**default_settings, **settings}
        else:
            self.settings = default_settings

        self.learning_rate = self.settings["learning_rate"]
        self.epochs = self.settings["epochs"]


        ### MAIN MCP
        # TODO: randomly initialize via rng
        self.weights = self.rng.uniform(-1, 1, size=(257, 10)) * self.settings["weight_scale"]
        self.bias = np.zeros(shape=(257, 10))
        self.__truth = np.zeros(10)
        self.activation_func = self.__argmax  # self.__step_func


    def fit(self, X, y):
        n_samples, n_features = X.shape

        _X = self.__append_bias(X)

        for __ in tqdm(np.arange(self.epochs)):
            for x, y_train in zip(_X, y):
                self.__truth[y_train] = 1.

                y_predicted = self.__predict(x)
                error = self.__truth - y_predicted

                update = self.learning_rate * error

                self.weights += np.outer(x, update) * self.__sigmoid_prime(np.dot(x, self.weights))
                self.bias += update

                self.__truth[y_train] = 0.



    def __predict(self, x):
        lin_out = np.dot(x, self.weights)  # X @ self.weights.T + self.bias
        y_predicted = self.__sigmoid(lin_out)
        return y_predicted

    def predict(self, X):
        _X = self.__append_bias(X)
        predictions = np.zeros(len(_X))
        for i, x in enumerate(_X):
            y = self.__predict(x)
            predictions[i] = np.argmax(y)

        return predictions


    def __step_func(self, x):
        return np.where(x >= 0, 1, 0)

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
            return np.mean(y_true == y_pred)
        else:
            raise NotImplementedError

def train_mc_perceptron():
    import matplotlib
    # matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import pandas as pd
    df_in = pd.read_csv('train_in.csv', header=None)
    train_in = df_in.to_numpy()

    df_out = pd.read_csv('train_out.csv', header=None)
    train_out = df_out.to_numpy()

    percy = MultiClassPerceptron()
    percy.fit(train_in, train_out)

    df_in = pd.read_csv('test_in.csv', header=None)
    test_in = df_in.to_numpy()

    df_out = pd.read_csv('test_out.csv', header=None)
    test_out = df_out.to_numpy()

    predictions = percy.predict(test_in)

    print(f"Accuracy: {percy.accuracy(test_out, predictions)}")


if __name__ == '__main__':
    train_mc_perceptron()