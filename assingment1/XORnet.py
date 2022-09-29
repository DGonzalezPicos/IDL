import numpy as np
from tqdm import tqdm

class XorNet(object):
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
        self.weights = self.rng.uniform(-1, 1, size=(3,3)) * self.settings["weight_scale"]
        self.bias = np.zeros(shape=(2))
        self.activation_func = self.__sigmoid

    def xor_net(self):
        pass

    def mse(self):
        pass

    def grdmse(self):
        pass

    @staticmethod
    def __sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def __sigmoid_prime(x):
        x = 1 / (1 + np.exp(-x))
        return x * (1. - x)


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
    ### input will be [0,1 ; 0,1]
    ### output should be prediction either logits or one hot
    pass