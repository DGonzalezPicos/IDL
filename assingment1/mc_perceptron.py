import numpy as np
import pandas as pd


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
        self.epochs = int(self.settings["epochs"])

        ### MAIN MCP
        # set initial weights, if they are initialized at w_i!=0 the first evaluation already steps them differently
        self.weights = self.rng.uniform(-1, 1, size=(257, 10)) * self.settings["weight_scale"]
        self.bias = np.zeros(shape=10)
        self.__truth = np.zeros(10)
        self.activation_func = self.__argmax

    def fit(self, X, y):
        # just the tip from the assignment sheet
        _X = self.__append_bias(X)

        for __ in np.arange(self.epochs):
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


def experiment(n_samples=50, verbose=True, settings={}):
    accuracies = np.zeros(n_samples)

    train_in, train_out, test_in, test_out = get_data()

    if verbose:
        _predictions = np.zeros((n_samples, len(test_out)))

    for i in range(n_samples):
        percy = MultiClassPerceptron(settings=settings)
        percy.fit(train_in, train_out)

        predictions = percy.predict(test_in)
        if verbose:
            _predictions[i] = predictions
        accuracies[i] = percy.accuracy(test_out, predictions)

    if verbose:
        try:
            from sklearn.metrics import ConfusionMatrixDisplay
            import matplotlib
            matplotlib.use('TkAgg')
            import matplotlib.pyplot as plt
            ConfusionMatrixDisplay.from_predictions(np.tile(test_out, reps=(n_samples, 1)).flatten(), _predictions.flatten(),
                                                    normalize="true", include_values=False)
            plt.title(f"MCP confusion matrix (N={n_samples})")
            plt.savefig("MCP_CFM.png", dpi=400)
            plt.show()
        except ModuleNotFoundError:
            pass

    if verbose:
        print(f"Accuracy (n={n_samples}):\n"
              f"\tMedian: {np.median(accuracies):.3f}, \n"
              f"\tMean: {np.mean(accuracies):.3f} +/- {np.std(accuracies):.5f} (1 SD)")
    return accuracies

def survey(n_grid=10, n_samples=25):
    # TODO: this ofc takes ages, probably worth to parallelize or vectorize to run on GPU, cupy
    train_in, train_out, test_in, test_out = get_data()

    # gridsearch
    l = np.geomspace(1e-6, 1e-1, n_grid)
    e = np.geomspace(10, 1000, n_grid)
    ll, ee = np.meshgrid(l, e)

    # this one is fully overwritten every data point, so it can just be initialized here
    accuracies = np.zeros(n_samples)

    mean_accuracy = np.zeros(ll.size)
    std_accuracy = np.zeros(ll.size)

    for i, (lr, n_epochs) in enumerate(zip(ll.flatten(), ee.flatten())):
        settings = {
            "learning_rate": lr,
            "epochs": n_epochs
        }

        for j in range(n_samples):
            percy = MultiClassPerceptron(settings=settings)
            percy.fit(train_in, train_out)

            predictions = percy.predict(test_in)
            accuracies[j] = percy.accuracy(test_out, predictions)

        accuracy = experiment(n_samples=n_samples, verbose=False)
        mean_accuracy[i] = np.mean(accuracy)
        std_accuracy[i] = np.std(accuracy)

    df_mean = pd.DataFrame(data=mean_accuracy.reshape(ll.shape), columns=l, index=e)
    df_mean.to_csv("mc_perceptron_mean_accuracy.csv")
    df_std = pd.DataFrame(data=mean_accuracy.reshape(ll.shape), columns=l, index=e)
    df_std.to_csv("mc_perceptron_std_accuracy.csv")

    plot_results()

def plot_results():
    df_mean = pd.read_csv("mc_perceptron_mean_accuracy.csv", index_col=0, header=0)

    lr = df_mean.columns.to_numpy().astype(float)
    epochs = df_mean.index.to_numpy().astype(float)

    data = df_mean.to_numpy().astype(float)

    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(constrained_layout=True)

    cf = ax.contourf(lr, epochs, data, cmap='RdGy')
    ax.set_xlim(np.min(lr), np.max(lr))
    ax.set_ylim(np.min(epochs), np.max(epochs))
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Total epochs")


    cb = plt.colorbar(cf, label="Mean accuracy (N=5)")

    plt.savefig("MCP_survey.png", dpi=400)

    plt.show()


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

def opt_MCP(n_samples=1):
    try:
        from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials

        trials = Trials()

        space = {
            "epochs": hp.quniform("epochs", 1,  100, 1),
            "learning_rate": hp.loguniform("learning_rate", np.log10(1.0e-10), np.log10(0.9)),
        }

        def objective(vector):
            settings = {
                "learning_rate": vector["learning_rate"],
                "epochs": vector["epochs"]
            }

            accuracies = experiment(n_samples=n_samples, settings=settings, verbose=False)

            mean_accuracy = np.mean(accuracies)

            return {'loss': 1 - mean_accuracy, 'status': STATUS_OK}

        tpe._default_n_startup_jobs = 25

        with np.errstate(under='ignore'):
            best_param = fmin(
                fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=100,
                trials=trials
            )

        import matplotlib
        matplotlib.use('TkAgg')
        from hyperopt.plotting import main_plot_history, main_plot_histogram, main_plot_vars

        main_plot_history(trials)
        main_plot_histogram(trials)
        main_plot_vars(trials)

        print(best_param)
    except ModuleNotFoundError:
        print("hyperpt module required for hyperparameter optimization")
        pass


if __name__ == '__main__':
    # __ = experiment(n_samples=10)
    # survey(n_grid=10, n_samples=5)
    # plot_results()
    opt_MCP()
