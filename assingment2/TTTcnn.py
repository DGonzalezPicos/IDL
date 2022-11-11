import numpy as np
import pandas as pd
import cupy as cp
import tensorflow as tf
# from tensorflow import keras
from tensorflow.keras.losses import Loss, CosineSimilarity
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Input
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from pathlib import Path
from functools import partial

global RNG
RNG = cp.random.default_rng(seed=None)

### MODELS
class RegressionCNN(Sequential):
    def __init__(self,
                 settings=None,
                 seed=None,
                 **kwargs):

        DefaultConv2D = partial(Conv2D, kernel_size=3, activation='leaky_relu', padding="VALID")
        stack = [
            Rescaling(1. / 255., input_shape=(150, 150, 1)),
            DefaultConv2D(filters=16, kernel_size=5),
            MaxPooling2D(pool_size=2),
            DefaultConv2D(filters=32),
            DefaultConv2D(filters=32),
            MaxPooling2D(pool_size=2),
            DefaultConv2D(filters=64),
            DefaultConv2D(filters=64),
            MaxPooling2D(pool_size=2),
            Dropout(0.4),
            Flatten(),
            Dense(units=512, activation='elu', kernel_initializer='he_normal'),
            Dense(units=512, activation='elu', kernel_initializer='he_normal'),
            Dense(units=2, activation='linear')
        ]

        super().__init__(stack)

        self.compile(optimizer="adam", loss=sincos_loss, metrics=['accuracy'])

        self.summary()

    def train(self, x, y, epochs=500, batchsize=128):
        self.fit(x, y,
                 epochs=epochs, batch_size=batchsize,
                 ) #validation_data=(dataset.x_valid, dataset.y_valid),)

    def test(self, x, y):
        print(self.evaluate(x, y, verbose=1))


class ClassificationCNN(object):
    pass


class TellTheTimeCNN(object):
    """
    Class for enslaving silicon to read analog watches.
    """
    def __init__(self,
                 settings=None,
                 seed=None,
                 **kwargs):
        super(TellTheTimeCNN).__init__()

        ### META

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

        ### CNN

    @staticmethod
    def time_error(yhat, y):
        pass


### LOSSES
# cosine similarity loss
cosine_similarity = CosineSimilarity(axis=-1)
def encode_sin_cos(y):
    return [
        [tf.math.sin(2 * np.pi * y[:, 0]), tf.math.cos(2 * np.pi * y[:, 0])],
        [tf.math.sin(2 * np.pi * y[:, 1]), tf.math.cos(2 * np.pi * y[:, 1])],
    ]

def sincos_loss(yhat, y):
    y_encoded = encode_sin_cos(y)
    yhat_encoded = encode_sin_cos(yhat)
    return cosine_similarity(y_encoded, yhat_encoded)

def new_sincos_loss(yhat, y, which=[0, 1], weights=[1., 1.]):
    loss = 0.
    for dim, weight in zip(which, weights):
        y_encoded = tf.map_fn(encode_sin_cos, y[:, dim], dtype=[tf.float32, tf.float32])
        yhat_encoded = tf.map_fn(encode_sin_cos, yhat[:, dim], dtype=[tf.float32, tf.float32])
        loss += weight * cosine_similarity(y_encoded, yhat_encoded) / sum(weights)

    return loss

# decimal loss
def encode_decimal(y): #stupid
    return y[:, 0] * 12 + y[:, 1]


### UTILS
def encode_time(y):
    """
    Returns the time in 2 pi units from proper time.
    :param y:
    :return:
    """
    y[:, 0] *= 1 / 12.
    y[:, 1] *= 1 / 60.
    return y

def decode_time(y):
    """
    Returns the proper time from time in 2 pi units.
    :param y:
    :return:
    """
    y[:, 0] *= 12.
    y[:, 1] *= 60.
    return y


def read_data(path="./a2_data"):
    # converts to f32
    path = Path(path)

    images = np.load(path / "images.npy")
    labels = np.load(path / "labels.npy")
    return tf.convert_to_tensor(images, dtype=tf.float32), tf.convert_to_tensor(labels, dtype=tf.float32)

def prepare_data(x, y, test_fraction=0.2):
    # y = encode_time(y)

    n_samples = len(y)
    n_train = int(n_samples * (1-test_fraction))

    idxs = tf.range(n_samples)
    idxs = tf.random.shuffle(idxs)

    train_idxs = idxs[:n_train]
    test_idxs = idxs[n_train:]

    x_train = tf.gather(x, indices=train_idxs)
    y_train = tf.gather(y, indices=train_idxs)

    x_test = tf.gather(x, indices=test_idxs)
    y_test = tf.gather(y, indices=test_idxs)

    return x_train, y_train, x_test, y_test

def get_data(path="./a2_data", test_fraction=0.2):

    images, labels = read_data(path=path)

    x_train, y_train, x_test, y_test = prepare_data(images, labels, test_fraction=test_fraction)

    assert x_train.device == y_train.device, "The data needs to be on the same device."
    print(f"Available devices: {tf.config.list_physical_devices('GPU')}")
    print(f"Training data on {x_train.device}")

    dataset = tf.data.Dataset.from_tensors((images, labels))

    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    print(x_train.shape)

    return x_train, y_train, x_test, y_test

if __name__ == '__main__':
    # y = tf.convert_to_tensor([
    #     [0.001, 0.95],
    #     [0.002, 0.96],
    #     [0.003, 0.97],
    #     [0.004, 0.98],
    #     [0.005, 0.99]
    # ], dtype=tf.float32)
    #
    # y_hat = tf.convert_to_tensor([
    #     [0.0011, 0.955],
    #     [0.0022, 0.966],
    #     [0.0033, 0.977],
    #     [0.0044, 0.988],
    #     [0.0055, 0.999]
    # ], dtype=tf.float32)[:,::-1]
    #
    # # y_hat = tf.convert_to_tensor([
    # #     [0.1, 0.5],
    # #     [0.2, 0.6],
    # #     [0.3, 0.7],
    # #     [0.4, 0.8],
    # #     [0.5, 0.9]
    # # ], dtype=tf.float32)
    #
    # print(y)
    # print(y_hat)
    # print(sincos_loss(y_hat, y))

    x_train, y_train, x_test, y_test = get_data()
    model = RegressionCNN()
    model.train(x_train, y_train)
    model.test(x_test, y_test)

    print(model.predict(x_test[:5]))
    print(y_test[:5])
