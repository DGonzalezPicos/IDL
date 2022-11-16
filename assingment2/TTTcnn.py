import numpy as np
import pandas as pd
import cupy as cp
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.losses import Loss, CosineSimilarity
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Input, SpatialDropout2D, LocallyConnected2D
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from pathlib import Path
from functools import partial

cosine_similarity = CosineSimilarity(axis=1)
mse = tf.keras.losses.MeanSquaredError()
mae = tf.keras.losses.MeanAbsoluteError()

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
        hour = tf.keras.activations.tanh(hour)
        # hour = tf.keras.activations.relu(x=hour, alpha=0.0, max_value=1.0, threshold=0.0)

        # Minute Branch
        minute = Dense(256, activation='leaky_relu')(x)
        minute = Dense(256, activation='leaky_relu')(minute)
        minute = Dense(1, activation="linear", name='minute')(minute)
        minute = tf.keras.activations.tanh(minute)
        # minute = tf.keras.activations.relu(x=minute, alpha=0.0, max_value=1.0, threshold=0.0)

        super().__init__(inputs=inputs, outputs=minute)

        self.compile(optimizer="adam", loss=[sincos_loss]) #, metrics=['accuracy'])
        self.compile(optimizer="adam", loss=['mse']) #, metrics=['mse', 'mae'])  # , metrics=['accuracy'])

        self.summary()

    def train(self, x, y, epochs=50, batchsize=128):
        self.fit(x, y,
                 epochs=epochs, batch_size=batchsize,
                 ) #validation_data=(dataset.x_valid, dataset.y_valid),)

    def test(self, x, y):
        print(self.evaluate(x, y, verbose=1))


class ClassificationCNN(object):
    pass


class TellTheTimeCNN(Model):
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
            "batch_size": 128,
            "encoding": "decimal",
            "type": ["regression"],  # classification, regression (can be sequence)
            "actfn_normalization": ["tanh"],  # must be sequence if type is sequence
            "loss": ["mse_sincos"]  # must be sequence if type is sequence
        }
        if isinstance(settings, dict):
            self.settings = {**default_settings, **settings}
        else:
            self.settings = default_settings

        __builders = {"regression": self.build_regression_head,
                      "classification": self.build_classification_head}
        __losses = {"mse_sincos": self.mse_sincos_loss,
                    "mse": "mse",
                    "categorical": 'categorical_crossentropy'}

        self.learning_rate = self.settings["learning_rate"]
        self.epochs = int(self.settings["epochs"])

        self.type = self.settings["type"]
        self.actfn_normalization = self.settings["actfn_normalization"]
        self.loss = self.settings["loss"]

        ### CNN
        KernelConv = partial(Conv2D, activation='leaky_relu', padding="VALID")
        LocalKernelConv = partial(LocallyConnected2D, activation='leaky_relu', padding="VALID")
        HalvingDropout = Dropout(.5)
        SpatialDroput = SpatialDropout2D(0.5)

        inputs = Input(shape=(150, 150, 1), name='input'),
        main_stack = [
            Rescaling(1. / 255., input_shape=(150, 150, 1)),
            KernelConv(filters=64, kernel_size=5, strides=2),
            MaxPooling2D(pool_size=2, strides=2),
            KernelConv(filters=64, kernel_size=3, strides=1),
            MaxPooling2D(pool_size=2, strides=1),
            KernelConv(filters=64, kernel_size=3, strides=1),
            MaxPooling2D(pool_size=2, strides=1),
            KernelConv(filters=64, kernel_size=3, strides=1),
            SpatialDroput,
            Flatten(),
        ]
        x = inputs
        for layer in main_stack:
            x = layer(x)

        outputs = []
        losses = []

        for i, (head, actfn, loss) in enumerate(zip(self.type, self.actfn_normalization, self.loss)):
            outputs.append(__builders[head](x=x, ))
            losses.append(__losses[loss])


        # build model
        super().__init__(inputs=inputs, outputs=outputs)
        self.compile(optimizer="adam", loss=losses)
        self.summary()


    def build_regression_head(self, x, name="reg_output"):
        regression_stack = [
            Dense(256, activation='leaky_relu'),
            Dense(256, activation='leaky_relu'),
            Dense(1, activation="tanh", name=name)
        ]
        for layer in regression_stack:
            x = layer(x)
        return x

    def build_classification_head(self, x, classes=12, name="class_output"):
        classification_stack = [
            Dense(256, activation='leaky_relu'),
            Dense(256, activation='leaky_relu'),
            Dense(units=classes, activation='softmax', name=name)
        ]
        for layer in classification_stack:
            x = layer(x)
        return x

    def build_random_preprocessing_stack(self):
        """
        Randomly preprocess the images before feeding into the model.
        While we can assume that the images are well shuffled and
        the clocks are at random orientations, angles with random reflections etc. we can make sure that our network
        does not overfit/latch on quirks of our input.
        Rational:
        Padding: Some transformations will otherwise potentially bring features out of the image.
        RandomRotation: the clocks are already rotated so adding to it should be ok.
        RandomContrast: the clocks have reflections and different light conditions.
        RandomBrightness: see above, but the version we use doesnt support it

        :return:
        """
        from tensorflow.keras.layers.experimental.preprocessing import RandomRotation, RandomContrast
        preprocessing_stack = [
            RandomRotation((-1, 1), fill_mode="constant", interpolation="bilinear", fill_value=0.0),  # full rotation
            tf.keras.layers.RandomContrast(0.05)  # every pixel can have 5% deviation from mean
        ]
        return preprocessing_stack

    # ENCODINGS
    @staticmethod
    def default_encoding(y):
        return y

    @staticmethod
    def encode_decimal(y, norm=1./12.):
        return y * norm

    @staticmethod
    def encode_common_decimal(y):
        return 1 / 12.0 * (y[:, 0] + y[:, 1] / 60.0)

    # DECODINGS
    @staticmethod
    def decode_decimal(y, norm=1./12.):
        return y / norm

    @staticmethod
    def decode_common_decimal(y):
        """
        Decode decimal back to hours, minutes using modulus.
        :param y: hours,minutes decimal
        :return:
        """
        r = y % 1
        return [(y-r) * 12., r * 60.]

    @staticmethod
    def mse_sincos_loss(yhat, y):
        """
        Loss for cyclic input in [0, 1). Input can have any real value as long as the period is scaled to 0,1.
        Get the sin and cos of the input, find the mse between labels and predictions and
        returns the sum of squares of the sin and cosine mse as the loss.
        :param yhat: predictions
        :param y: labels (truth)
        :return: loss, L > 0.
        """
        y_encoded_sin = encode_sin(y)
        yhat_encoded_sin = encode_sin(yhat)

        y_encoded_cos = encode_cos(y)
        yhat_encoded_cos = encode_cos(yhat)

        sin_loss = mse(y_encoded_sin, yhat_encoded_sin)
        cos_loss = mse(y_encoded_cos, yhat_encoded_cos)

        loss = tf.square(sin_loss) + tf.square(cos_loss)
        return loss


### LOSSES
# cosine similarity loss


def encode_sin_cos(y):
    return [tf.math.sin(2 * np.pi * y), tf.math.cos(2 * np.pi * y)]

def encode_sin(y):
    return tf.math.sin(2 * np.pi * y)

def encode_cos(y):
    return tf.math.cos(2 * np.pi * y)

def mse_sincos_loss(yhat, y):
    # y_encoded = encode_sin_cos(y)
    # yhat_encoded = encode_sin_cos(yhat)

    y_encoded_sin = encode_sin(y)
    yhat_encoded_sin = encode_sin(yhat)

    y_encoded_cos = encode_cos(y)
    yhat_encoded_cos = encode_cos(yhat)

    sin_loss = mse(y_encoded_sin, yhat_encoded_sin)
    cos_loss = mse(y_encoded_cos, yhat_encoded_cos)

    loss = tf.math.sqrt(tf.square(sin_loss) + tf.square(cos_loss))
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

    # dataset = tf.data.Dataset.from_tensors((images, labels))

    return x_train, y_train, x_test, y_test

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = get_data()
    y_train, y_test = encode_decimal(y_train), encode_decimal(y_test)

    model = RegressionCNN()
    model.train(x_train, y_train)
    model.test(x_test, y_test)
