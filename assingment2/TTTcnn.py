import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.losses import Loss, CosineSimilarity
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Input, SpatialDropout2D, \
    LocallyConnected2D, LayerNormalization, BatchNormalization, Concatenate
from tensorflow.keras.layers.experimental import EinsumDense
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from pathlib import Path
from functools import partial

cosine_similarity = CosineSimilarity(axis=1)
mse = tf.keras.losses.MeanSquaredError()
mae = tf.keras.losses.MeanAbsoluteError()
hubby = tf.keras.losses.Huber(delta=0.5)


### MODEL
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
        tf.random.set_seed(seed)

        ### DEFAULT VALUES
        default_settings = {
            "learning_rate": 1e-2,
            "epochs": 50,
            "batch_size": 128,
            "encoding": "decimal",
            "type": ["regression"],             # classification, regression (can be sequence)
            "actfn_normalization": ["tanh"],    # must be sequence if type is sequence
            "loss": ["mae_angle"],              # must be sequence if type is sequence
            "n_classes": [12],                  # must be sequence if type is sequence
        }
        if isinstance(settings, dict):
            self.settings = {**default_settings, **settings}
        else:
            self.settings = default_settings

        __builders = {"regression": self.build_regression_head,
                      "classification": self.build_classification_head}
        __losses = {"mae_angle": self.mae_angle_loss,
                    "linear_decimal_cyclic": self.linear_cyclic_loss,
                    "mse": "mse",
                    "categorical": 'categorical_crossentropy'}
        __encodings = {"decimal": self.encode_common_decimal,
                       }

        self.learning_rate = self.settings["learning_rate"]
        self.epochs = int(self.settings["epochs"])

        self.encoding = self.settings["encoding"]
        self.encoding_fn = __encodings[self.encoding]
        self.type = self.settings["type"]
        self.actfn_normalization = self.settings["actfn_normalization"]
        self.loss = self.settings["loss"]
        self.n_classes = self.settings["n_classes"]

        ### CNN
        # KernelConv = partial(Conv2D, activation=hidden_actfn, kernel_initializer="he_normal", padding="VALID")
        # LocalKernelConv = partial(LocallyConnected2D, activation=hidden_actfn, padding="VALID")
        # HalvingDropout = Dropout(.4)
        # SpatialDroput = SpatialDropout2D(0.4)
        hidden_actfn = "leaky_relu"
        

        inputs = Input(shape=(150, 150, 1), name='input')
        main_stack = [
            Rescaling(1. / 255., input_shape=(150, 150, 1)),

            Conv2D(filters=64, kernel_size=3, strides=1,
                   activation=hidden_actfn, kernel_initializer="he_normal", padding="VALID"),
            Conv2D(filters=64, kernel_size=3, strides=1,
                   activation=hidden_actfn, kernel_initializer="he_normal", padding="VALID"),
            BatchNormalization(axis=1),
            MaxPooling2D(pool_size=2),

            Conv2D(filters=32, kernel_size=3, strides=1,
                   activation=hidden_actfn, kernel_initializer="he_normal", padding="VALID"),
            Conv2D(filters=32, kernel_size=3, strides=1,
                   activation=hidden_actfn, kernel_initializer="he_normal", padding="VALID"),
            BatchNormalization(axis=1),
            MaxPooling2D(pool_size=2),

            Conv2D(filters=16, kernel_size=5, strides=1,
                   activation=hidden_actfn, kernel_initializer="he_normal", padding="VALID"),
            BatchNormalization(axis=1),
            MaxPooling2D(pool_size=2, strides=1),

            Conv2D(filters=32, kernel_size=3, strides=1,
                   activation=hidden_actfn, kernel_initializer="he_normal", padding="VALID"),
            Conv2D(filters=32, kernel_size=3, strides=1,
                   activation=hidden_actfn, kernel_initializer="he_normal", padding="VALID"),
            BatchNormalization(axis=1),
            MaxPooling2D(pool_size=2),

            Conv2D(filters=64, kernel_size=3, strides=1,
                   activation=hidden_actfn, kernel_initializer="he_normal", padding="VALID"),
            Conv2D(filters=64, kernel_size=3, strides=1,
                   activation=hidden_actfn, kernel_initializer="he_normal", padding="VALID"),
            BatchNormalization(axis=1),
            MaxPooling2D(pool_size=2),

            # Dropout(.4),
            Flatten(),
        ]
        x = inputs
        for layer in main_stack:
            x = layer(x)

        outputs = []
        losses = []

        for i, (head, actfn, loss, n_classes) in enumerate(zip(self.type,
                                                               self.actfn_normalization,
                                                               self.loss,
                                                               self.n_classes)):
            outputs.append(__builders[head](_x=x, classes=n_classes))
            losses.append(__losses[loss])

        # build model
        super().__init__(inputs=inputs, outputs=outputs)
        self.compile(optimizer="adam", loss="mae") #losses)  #
        self.summary()

    def build_regression_head(self, _x, name="reg_output", **kwargs):
        regression_stack = [
            Dense(1024, activation="leaky_relu", kernel_initializer='he_normal'),
            BatchNormalization(),
            Dense(256, activation="leaky_relu", kernel_initializer='he_normal'),
            BatchNormalization(),
            Dense(32, activation="leaky_relu", kernel_initializer='he_normal'),
            BatchNormalization(),
            # Dense(1, activation="tanh"),
            Dense(1, activation="linear", name=name)
        ]

        for layer in regression_stack:
            _x = layer(_x)
        return _x

    def build_classification_head(self, _x, classes=12, name="class_output", **kwargs):
        classification_stack = [
            Dense(256, activation="relu"),
            Dense(256, activation="relu"),
            Dense(units=classes, activation='softmax', name=name)
        ]
        for layer in classification_stack:
            _x = layer(_x)
        return _x

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
    def encode_decimal(y, norm=1. / 12.):
        return y * norm

    @staticmethod
    def encode_common_decimal(y):
        return 1 / 12.0 * (y[:, 0] + y[:, 1] / 60.0)

    @staticmethod
    def encode_common_classes(y, n_classes):
        """
        Class encoding using bankers rounding. Encodes from hh:mm to linear [0, 1) and then to classes
        with labels 0, 1... n_classes - 1 with "increasing" time.
        This means this can also be used for regression or with the sin-cos losses etc.
        :param y:
        :param n_classes:
        :return:
        """
        linear = 1 / 12.0 * (y[:, 0] + y[:, 1] / 60.0)
        linear *= n_classes
        return tf.round(linear)

    # DECODINGS
    @staticmethod
    def decode_decimal(y, norm=1. / 12.):
        return y / norm

    @staticmethod
    def decode_common_decimal(y):
        """
        Decode decimal back to hours, minutes using modulus.
        :param y: hours,minutes decimal
        :return:
        """
        r = y % 1
        return [(y - r) * 12., r * 60.]

    @staticmethod
    def mae_angle_loss(y, yhat):
        """
        Loss for cyclic input in [0, 1). Input can have any real value as long as the period is scaled to 0,1.
        Get the sin and cos of the input, find the mse between labels and predictions and
        returns the sum of squares of the sin and cosine mse as the loss.
        :param yhat: predictions
        :param y: labels (truth)
        :return: loss, L > 0.
        """
        y_encoded_sin = tf.math.sin(2 * np.pi * y)
        yhat_encoded_sin = tf.math.sin(2 * np.pi * yhat)

        y_encoded_cos = tf.math.cos(2 * np.pi * y)
        yhat_encoded_cos = tf.math.cos(2 * np.pi * yhat)

        sin_loss = mae(y_encoded_sin, yhat_encoded_sin)
        cos_loss = mae(y_encoded_cos, yhat_encoded_cos)

        # sin_loss = tf.math.square(y_encoded_sin - yhat_encoded_sin)
        # cos_loss = tf.math.square(y_encoded_cos - yhat_encoded_cos)

        loss = tf.sqrt(tf.square(sin_loss) + tf.square(cos_loss))
        return loss

    @staticmethod
    def mse_sincos_loss(y, yhat):
        """
        Loss for cyclic input in [0, 1). Input can have any real value as long as the period is scaled to 0,1.
        Get the sin and cos of the input, find the mse between labels and predictions and
        returns the sum of squares of the sin and cosine mse as the loss.
        :param yhat: predictions
        :param y: labels (truth)
        :return: loss, L > 0.
        """
        y_encoded_sin = tf.math.sin(2 * np.pi * y)
        yhat_encoded_sin = tf.math.sin(2 * np.pi * yhat)

        y_encoded_cos = tf.math.cos(2 * np.pi * y)
        yhat_encoded_cos = tf.math.cos(2 * np.pi * yhat)

        sin_loss = mae(y_encoded_sin, yhat_encoded_sin)
        cos_loss = mae(y_encoded_cos, yhat_encoded_cos)

        # sin_loss = tf.math.square(y_encoded_sin - yhat_encoded_sin)
        # cos_loss = tf.math.square(y_encoded_cos - yhat_encoded_cos)

        loss = tf.sqrt(tf.square(sin_loss) + tf.square(cos_loss))
        return loss

    @staticmethod
    def linear_cyclic_loss(y, yhat):
        loss = tf.math.minimum((yhat + 1. - y) % 1, (y + 1 - yhat) % 1)
        return tf.square(loss)

    def encode_y(self, y):
        return self.encoding_fn(y)

    def train(self, x, y, epochs=4, batchsize=128, shuffle=True, validation_data=None, validation_freq=None):
        if validation_data is None and validation_freq is not None:
            validation_freq = int(0.2 * epochs)

        history = self.fit(x, y,
                           epochs=epochs, batch_size=batchsize,
                           validation_data=validation_data, validation_freq=validation_freq,
                           shuffle=shuffle,)

    def test(self, x, y):
        eval = self.evaluate(x, y, verbose=1, return_dict=True)
        print(eval)


### UTILS
def read_data(path="./a2_data"):
    # converts to f32
    path = Path(path)

    images = np.load(path / "images.npy")
    labels = np.load(path / "labels.npy")
    return tf.convert_to_tensor(images, dtype=tf.float32), tf.convert_to_tensor(labels, dtype=tf.float32)


def prepare_data(x, y, test_fraction=0.2):
    # y = encode_time(y)

    n_samples = len(y)
    n_train = int(n_samples * (1 - test_fraction))

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
    # print(f"Available devices: {tf.config.list_physical_devices('GPU')}")
    print(f"Training data on {x_train.device}")

    # dataset = tf.data.Dataset.from_tensors((images, labels))

    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    # print("No longer supported, use the supplied jupyter notebook.")
    x_train, base_y_train, x_test, base_y_test = get_data()
    default_model = TellTheTimeCNN()

    print(base_y_train.shape)
    print(x_train.shape)

    y_train, y_test = default_model.encode_y(base_y_train), default_model.encode_y(base_y_test)

    try:
        print("Encoding from hh,mm -> f: ", base_y_train.shape, " -> ", y_train.shape)
    except AttributeError:
        print("Encoding from hh,mm -> f: ", base_y_train.shape, " -> ", len(y_train))


    # import matplotlib
    # matplotlib.use('TkAgg')
    # import matplotlib.pyplot as plt
    #
    # fig, (ax, ax2, ax3, ax4) = plt.subplots(1, 4, constrained_layout=True, figsize=(24, 6))
    # ax.hist(y_train, bins=100)
    # ax.hist(y_test, bins=100)
    # hist = ax2.hist2d(base_y_train[:, 0], base_y_train[:, 1], vmin=0., bins=100)
    # fig.colorbar(hist[3], ax=ax2)
    # hist = ax3.hist2d(base_y_test[:, 0], base_y_test[:, 1], vmin=0., bins=100)
    # fig.colorbar(hist[3], ax=ax3)
    # ax4.hist(base_y_train[:, 1], bins=100)
    # plt.show()

    # yhat, y = np.linspace(-1, 1., 200), np.linspace(-1, 1., 200)
    # yyhat, yy = np.meshgrid(yhat, y)
    #
    # def sincos(y, yhat):
    #     """
    #     Loss for cyclic input in [0, 1). Input can have any real value as long as the period is scaled to 0,1.
    #     Get the sin and cos of the input, find the mse between labels and predictions and
    #     returns the sum of squares of the sin and cosine mse as the loss.
    #     :param yhat: predictions
    #     :param y: labels (truth)
    #     :return: loss, L > 0.
    #     """
    #     y_encoded_sin = np.sin(2 * np.pi * y)
    #     yhat_encoded_sin = np.sin(2 * np.pi * yhat)
    #
    #     y_encoded_cos = np.cos(2 * np.pi * y)
    #     yhat_encoded_cos = np.cos(2 * np.pi * yhat)
    #
    #     sin_loss = np.square(yhat_encoded_sin - y_encoded_sin)
    #     cos_loss = np.square(yhat_encoded_cos - y_encoded_cos)
    #
    #     loss = np.sqrt(np.square(sin_loss) + np.square(cos_loss))
    #     return loss
    #
    # loss = sincos(yy.flatten(), yyhat.flatten())
    #
    # import matplotlib
    # matplotlib.use('TkAgg')
    # import matplotlib.pyplot as plt
    #
    # plt.matshow(loss.reshape(200, 200))
    # plt.show()
    #
    # raise NotImplementedError

    default_model.train(x_train, y_train)
    default_model.test(x_test, y_test)

    print(y_test[:5])
    print(default_model.predict(x_test[:5]))

    print("=============")

    print(y_train[:5])
    print(default_model.predict(x_train[:5]))
