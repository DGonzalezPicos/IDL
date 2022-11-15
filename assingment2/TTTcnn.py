import numpy as np
import pandas as pd
import cupy as cp
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.losses import Loss, CosineSimilarity
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Input
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from pathlib import Path
from functools import partial

# tf.config.run_functions_eagerly(False)

global RNG
RNG = cp.random.default_rng(seed=None)

### MODELS
class RegressionCNN(Model):
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

        inputs = tf.keras.Input(shape=(150, 150, 1), name='input')
        x = Rescaling(1. / 255., input_shape=(150, 150, 1))(inputs)
        x = Conv2D(64, kernel_size=5, strides=2, activation='leaky_relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
        x = Conv2D(64, kernel_size=3, strides=1, activation='leaky_relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(64, kernel_size=3, strides=1, activation='leaky_relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(64, kernel_size=3, strides=1, activation='leaky_relu')(x)
        x = Dropout(.4)(x)
        x = Flatten()(x)

        # Hour branch
        hour = Dense(256, activation='leaky_relu')(x)
        hour = Dense(256, activation='leaky_relu')(hour)
        hour = Dense(1, activation="linear", name='hour')(hour)
        # hour = tf.keras.activations.relu(x=hour, alpha=0.0, max_value=1.0, threshold=0.0)

        # Minute Branch
        minute = Dense(256, activation='leaky_relu')(x)
        minute = Dense(256, activation='leaky_relu')(minute)
        minute = Dense(1, activation="linear", name='minute')(minute)
        # minute = tf.keras.activations.relu(x=minute, alpha=0.0, max_value=1.0, threshold=0.0)

        super().__init__(inputs=inputs, outputs=minute)

        self.compile(optimizer="adam", loss=sincos_loss) #, metrics=['accuracy'])
        # self.compile(optimizer="adam", loss=['mse', 'mse']) #, metrics=['mse', 'mae'])  # , metrics=['accuracy'])

        self.summary()

    def train(self, x, y, epochs=10, batchsize=128):
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
cosine_similarity = CosineSimilarity(axis=1)
mse = tf.keras.losses.MeanSquaredError()
mae = tf.keras.losses.MeanAbsoluteError()

def encode_sin_cos(y):
    return [tf.math.sin(2 * np.pi * y), tf.math.cos(2 * np.pi * y)]

def encode_sin(y):
    return tf.math.sin(2 * np.pi * y)

def encode_cos(y):
    return tf.math.cos(2 * np.pi * y)

def sincos_loss(yhat, y):
    # y_encoded = encode_sin_cos(y)
    # yhat_encoded = encode_sin_cos(yhat)

    y_encoded_sin = encode_sin(y)
    yhat_encoded_sin = encode_sin(yhat)

    y_encoded_cos = encode_cos(y)
    yhat_encoded_cos = encode_cos(yhat)

    sin_loss = mse(yhat_encoded_sin, y_encoded_sin)
    cos_loss = mse(yhat_encoded_cos, y_encoded_cos)

    loss = tf.math.sqrt(tf.square(sin_loss) + tf.square(cos_loss))

    # print(yhat_encoded.shape)
    # loss = mae(yhat_encoded, y_encoded)
    # loss = cosine_similarity(2 * np.pi * y, 2 * np.pi * yhat)
    # loss = 1 - tf.math.sqrt()
    return loss

# decimal loss
def encode_decimal(y):
    return 1/12.0 * (y[:, 0] + y[:, 1] / 60.0)


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

def split_y(y):
    return [y[:, 0], y[:, 1]]

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
    x_train, y_train, x_test, y_test = get_data()
    # y_train, y_test = encode_time(y_train), encode_time(y_test)
    y_train, y_test = encode_decimal(y_train), encode_decimal(y_test)

    # raise NotImplementedError


    model = RegressionCNN()
    model.train(x_train, y_train)
    model.test(x_test, y_test)

    print(model.predict(x_test[:5]))
    print(y_test[:5])
