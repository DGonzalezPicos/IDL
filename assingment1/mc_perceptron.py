import numpy as np

class MultiClassPerceptron(object):
    def __init__(self,
                 settings=None, **kwargs):

        ### META
        self.rng = np.random.default_rng()

        ### DEFAULT VALUES
        default_settings = {
            "learning_rate": 1e-5,
            "epochs": 1000,
            "weight_scale": 1e-4
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
        self.activation_func = self.__argmax  # self.__step_func


    def fit(self, X, y):
        n_samples, n_features = X.shape

        _X = self.__append_bias(X)

        # TODO: this is the current sharpening, either 1 or 0
        _y = np.copy(y)
        _y[y > 0.] = 1
        _y[y <= 0.] = 0

        for __ in np.arange(self.epochs):
            for x, y_train in zip(_X, _y):
                y_predicted = self.predict(x)
                error = y_train - y_predicted

                update = self.learning_rate * error
                self.weights += update * x
                self.bias += update


    def predict(self, X):
        lin_out = X @ self.weights.T + self.bias
        y_predicted = self.activation_func(lin_out)
        return y_predicted

    def __step_func(self, x):
        return np.where(x >= 0, 1, 0)

    def __argmax(self, x):
        return np.argmax(x)

    @staticmethod
    def __append_bias(X):
        bias = np.ones(shape=(len(X), 1))
        return np.concatenate((X, bias), axis=1)

    @staticmethod
    def accuracy(y_true, y_pred=None):
        if y_pred is not None:
            return np.mean(y_true == y_pred)

if __name__ == '__main__':
    pass