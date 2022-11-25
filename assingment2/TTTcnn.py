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
from CNN_stacks import *

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
            "learning_rate": 1.0e-3,
            "final_learning_rate": 1.0e-7,
            "epochs": 50,
            "batch_size": 128,
            "encoding": "decimal",
            "type": ["regression"],             # classification, regression (can be sequence)
            "actfn_normalization": ["linear"],    # must be sequence if type is sequence
            "loss": ["mae_angle"],              # must be sequence if type is sequence
            "n_classes": [12],                  # must be sequence if type is sequence
            "scheduler": True,
            "decay": 5.0e-4,
            "main_stack": get_simple_classification_CNN(),
            "n_outputs": [1],                   # must be sequence if type is sequence
            "metric": "linear"
        }
        if isinstance(settings, dict):
            self.settings = {**default_settings, **settings}
        else:
            self.settings = default_settings

        __builders = {"regression": self.build_regression_head,
                      "classification": self.build_classification_head}
        __losses = {"mae_angle": self.mae_angle_loss,
                    "mse_sincos": self.mse_sincos_loss,
                    "linear_decimal_cyclic": self.linear_cyclic_loss,
                    "mse": "mse",
                    "mae": "mae",
                    "categorical": 'categorical_crossentropy',
                    "2out_regression": self.mse_2h_sincos_loss}
        __encodings = {"decimal": self.encode_common_decimal,
                       "double_decimal": self.encode_double_decimal,
                       "large_decimal": self.encode_large_decimal,
                       "common_classes": self.encode_common_classes_1h,
                       "hh_mm_classes": self.encode_hh_mm_classes,
                       "sin_cos": self.encode_sin_cos}
        __metrics = {"linear": self.metric_linear_cyclic,
                     "classes": ["accuracy", self._class_difference],
                     "2head": "accuracy"}

        self.learning_rate = self.settings["learning_rate"]
        self.final_learning_rate = self.settings["final_learning_rate"]
        self.scheduler = self.settings["scheduler"]
        self.decay = self.settings["decay"]
        self.epochs = int(self.settings["epochs"])

        self.encoding = self.settings["encoding"]
        self.encoding_fn = __encodings[self.encoding]
        self.type = self.settings["type"]
        self.actfn_normalization = self.settings["actfn_normalization"]
        self.loss = self.settings["loss"]
        print(self.settings["metric"])
        self.metric = __metrics[self.settings["metric"]]
        self.n_classes = self.settings["n_classes"]
        self.n_outputs = self.settings["n_outputs"]
        if len(self.n_outputs) != len(self.type):
            self.n_outputs = np.full_like(self.type, fill_value=self.n_outputs[0], dtype=int)

        ### CNN
        # KernelConv = partial(Conv2D, activation=hidden_actfn, kernel_initializer="he_normal", padding="VALID")
        # LocalKernelConv = partial(LocallyConnected2D, activation=hidden_actfn, padding="VALID")
        # HalvingDropout = Dropout(.4)
        # SpatialDroput = SpatialDropout2D(0.4)
        hidden_actfn = "leaky_relu"
        

        inputs = Input(shape=(150, 150, 1), name='input')

        main_stack = self.settings["main_stack"]

        x = inputs
        for layer in main_stack:
            x = layer(x)

        outputs = []
        output_stacks = []
        losses = []
        # metrics = []

        for i, (head, actfn, loss, n_classes, n_output) in enumerate(zip(self.type,
                                                               self.actfn_normalization,
                                                               self.loss,
                                                               self.n_classes,
                                                                         self.n_outputs)):
            output, out_stack, out_name = __builders[head](_x=x, i=i, classes=n_classes, name=f"{i}", outs=n_output)
            outputs.append(output)
            output_stacks.append(out_stack)
            losses.append(__losses[loss])

        # metric = [{"accuracy": self.metric}]

        # build model
        # print(outputs)
        # print(losses)
        # print(self.metric)

        super().__init__(inputs=inputs, outputs=outputs)

        if self.scheduler:
            # self.learning_rate_fn = tf.keras.optimizers.schedules.CosineDecayRestarts(
            #     self.learning_rate, 2000, t_mul=2.0, m_mul=0.5, alpha=self.final_learning_rate,
            #     name="CosineDecayRestarts")
            self.learning_rate_fn = tf.keras.optimizers.schedules.ExponentialDecay(self.learning_rate,
                                                                                   1, 1 - self.decay, staircase=False,
                                                                                   name="ExponentialDecay")
        else:
            self.learning_rate_fn = self.learning_rate

        self.opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate_fn)
        self.compile(optimizer=self.opt, metrics=self.metric, loss=losses)  #"mae") #
        self.summary()

    def build_regression_head(self, _x, i=0, outs=1, name="DEFAULT", **kwargs):
        name = "reg_output_" + name
        regression_stack = [
            Dense(256, activation="leaky_relu", kernel_initializer='he_normal'),
            BatchNormalization(),
            Dense(64, activation="leaky_relu", kernel_initializer='he_normal'),
            Dense(outs, activation=self.actfn_normalization[i], name=name),
            Rescaling(scale=1. + 5.0e-1, offset=-1.0e-3),
        ]

        for layer in regression_stack:
            _x = layer(_x)
        return _x, regression_stack, name

    def build_classification_head(self, _x, classes=12, name="DEFAULT", **kwargs):
        name = "class_output_" + name
        classification_stack = [
            Dense(64, activation="leaky_relu"),
            Dense(64, activation="leaky_relu"),
            Dense(units=classes, activation='softmax', name=name)
        ]
        for layer in classification_stack:
            _x = layer(_x)
        return _x, classification_stack, name

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

    def build_graph(self, raw_shape):
        x = tf.keras.layers.Input(shape=(None, raw_shape),
                                         ragged=True)
        return tf.keras.Model(inputs=[x],
                              outputs=self.call(x))

    # ENCODINGS
    @staticmethod
    def default_encoding(y, **kwargs):
        return y

    @staticmethod
    def encode_decimal(y, norm=1. / 12., **kwargs):
        return y * norm

    @staticmethod
    def encode_common_decimal(y, **kwargs):
        return 1 / 12.0 * (y[:, 0] + y[:, 1] / 60.0)

    @staticmethod
    def encode_double_decimal(y, **kwargs):
        hh, mm = tf.unstack(y, axis=1)
        hh *= 1./12.
        mm *= 1. / 60.
        y = tf.stack([hh, mm], axis=1)
        return [hh, mm]  # y

    @staticmethod
    def encode_large_decimal(y, **kwargs):
        return y[:, 0] + y[:, 1] / 60.0

    def encode_common_classes(self, y, n_classes, **kwargs):
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
        linear = tf.cast(tf.round(linear), tf.uint8)
        return linear

    def encode_common_classes_1h(self, y, n_classes, **kwargs):
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
        linear = tf.cast(tf.round(linear), tf.uint8)
        return tf.one_hot(linear, n_classes)

    def encode_classes_1h(self, y, n_classes, **kwargs):
        """
        Class encoding using bankers rounding. Encodes from hh:mm to linear [0, 1) and then to classes
        with labels 0, 1... n_classes - 1 with "increasing" time.
        This means this can also be used for regression or with the sin-cos losses etc.
        :param y:
        :param n_classes:
        :return:
        """
        linear = y / tf.math.reduce_max(y)
        linear *= n_classes
        linear = tf.cast(tf.round(linear), tf.uint8)
        return tf.one_hot(linear, n_classes)

    def encode_classes(self, y, n_classes, **kwargs):
        """
        Class encoding using bankers rounding. Encodes from hh:mm to linear [0, 1) and then to classes
        with labels 0, 1... n_classes - 1 with "increasing" time.
        This means this can also be used for regression or with the sin-cos losses etc.
        :param y:
        :param n_classes:
        :return:
        """
        linear = y / tf.math.reduce_max(y)
        linear *= n_classes
        linear = tf.cast(tf.round(linear), tf.uint8)
        return linear

    def encode_hh_mm_classes(self, y, n_hh_classes, n_mm_classes, **kwargs):
        hh, mm = tf.unstack(y, axis=1)

        hh = self.encode_classes(hh, n_hh_classes)
        mm = self.encode_classes(mm, n_mm_classes)

        y = tf.stack([hh, mm], axis=1)
        # y = [hh, mm]

        return y

    def encode_sin_cos(self, y, n_outs=4, **kwargs):
        hh, mm = tf.unstack(y, axis=1)
        hh = self.encode_decimal(hh, norm=1./12.)
        mm = self.encode_decimal(mm, norm=1./60.)

        hh_encoded_sin = tf.math.sin(2 * np.pi * hh)
        hh_encoded_cos = tf.math.cos(2 * np.pi * hh)

        mm_encoded_sin = tf.math.sin(2 * np.pi * mm)
        mm_encoded_cos = tf.math.cos(2 * np.pi * mm)

        if n_outs == 2:
            hh = tf.stack([hh_encoded_sin, hh_encoded_cos], axis=-1)
            mm = tf.stack([mm_encoded_sin, mm_encoded_cos], axis=-1)
            y = tf.stack([hh, mm], axis=-1)
        else:
            y = tf.stack([hh_encoded_sin, hh_encoded_cos, mm_encoded_sin, mm_encoded_cos], axis=1)



        return y # [hh_encoded_sin, hh_encoded_cos, mm_encoded_sin, mm_encoded_cos] # y

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

        sin_loss = mse(y_encoded_sin, yhat_encoded_sin)
        cos_loss = mse(y_encoded_cos, yhat_encoded_cos)

        # sin_loss = tf.math.square(y_encoded_sin - yhat_encoded_sin)
        # cos_loss = tf.math.square(y_encoded_cos - yhat_encoded_cos)

        loss = sin_loss + cos_loss
        return loss

    @staticmethod
    def mse_2h_sincos_loss(y, yhat):
        """
        Loss for cyclic input in [0, 1). Input can have any real value as long as the period is scaled to 0,1.
        Get the sin and cos of the input, find the mse between labels and predictions and
        returns the sum of squares of the sin and cosine mse as the loss.
        :param yhat: predictions
        :param y: labels (truth)
        :return: loss, L > 0.
        """
        # y_encoded_sin = tf.math.sin(2 * np.pi * y)
        # y_encoded_cos = tf.math.cos(2 * np.pi * y)
        #
        # sin_loss = mse(y_encoded_sin, yhat[:, 0])
        # cos_loss = mse(y_encoded_cos, yhat[:, 1])

        sin_loss = mse(y[:, 0], yhat[:, 0])
        cos_loss = mse(y[:, 1], yhat[:, 1])

        # sin_loss = tf.math.square(y_encoded_sin - yhat_encoded_sin)
        # cos_loss = tf.math.square(y_encoded_cos - yhat_encoded_cos)

        # loss = sin_loss + cos_loss
        return [sin_loss, cos_loss]  # loss

    @staticmethod
    def linear_cyclic_loss(y, yhat):
        loss = tf.minimum(tf.abs(y - yhat), tf.abs(1 + y - yhat))
        return tf.square(loss)

    def metric_linear_cyclic(self, y, yhat):
        error = tf.minimum(tf.abs(y - yhat), tf.abs(1 + y - yhat))
        error *= 12 * 60
        return error


    def encode_y(self, y, **kwargs):
        return self.encoding_fn(y, **kwargs)

    def _predict_class(self, x):
        yhat = self.predict(x)
        yhat = tf.argmax(yhat, axis=-1)
        return yhat

    def _predict_4head_reg_labels(self, x):
        yhat = self.predict(x)

        try:
            hh_sin, hh_cos, mm_sin, mm_cos = yhat
        except BaseException:
            hh, mm = yhat
            hh_sin, hh_cos = tf.unstack(hh, axis=-1)
            mm_sin, mm_cos = tf.unstack(mm, axis=-1)

        hh_angle = (tf.math.atan2(hh_cos, hh_sin) + np.pi) / (2. * np.pi)
        mm_angle = (tf.math.atan2(mm_cos, mm_sin) + np.pi) / (2. * np.pi)

        hh_hours = hh_angle * 12.
        mm_minutes = mm_angle * 60.

        return tf.stack([hh_hours, mm_minutes], axis=1)

    def _class_difference(self, y, yhat, n_classes=None):
        if n_classes is None:
            n_classes = len(y[0])
        y = tf.argmax(y, axis=-1) / n_classes
        yhat = tf.argmax(yhat, axis=-1) / n_classes

        diff = tf.minimum(tf.abs(y - yhat), tf.abs(1 + y - yhat))
        return diff

    def train(self, x, y, epochs=100, batchsize=128, shuffle=True, validation_data=None, validation_freq=None):
        history = self.fit(x, y,
                           epochs=epochs, batch_size=batchsize,
                           validation_data=validation_data, # validation_freq=validation_freq,
                           shuffle=shuffle,
                           callbacks=[tf.keras.callbacks.EarlyStopping(patience=10),
                                      # tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                      #                                      patience=5, min_lr=0.0)
        ])  # TODO: doesnt work with LrScheduler, see issue https://github.com/tensorflow/tensorflow/issues/41639
            # which is still unresolved
        return history

    def test(self, x, y):
        eval = self.evaluate(x, y, verbose=1, return_dict=True)
        print(eval)
        return eval



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

    # settings = {
    #     "learning_rate": 1.0e-1,
    #     "encoding": "sin_cos",
    #     "type": ["regression", "regression", "regression", "regression"],  # classification, regression (can be sequence)
    #     "actfn_normalization": ["tanh", "tanh", "tanh", "tanh"],  # must be sequence if type is sequence
    #     "loss": ["mse", "mse", "mse", "mse"],  # must be sequence if type is sequence
    #     "n_classes": [12, 6, 1, 1],  # must be sequence if type is sequence
    #     "decay": 5.0e-10,
    #     "scheduler": False
    # }
    settings = {
        "learning_rate": 1.0e-3,
        "encoding": "common_classes",
        "type": ["classification"],  # classification, regression (can be sequence)
        "actfn_normalization": ["tanh"],  # must be sequence if type is sequence
        "loss": ["categorical"],  # must be sequence if type is sequence
        "n_classes": [72],  # must be sequence if type is sequence
        "decay": 5.0e-5,
        "scheduler": True
    }


    default_model = TellTheTimeCNN(settings=settings)

    tf.keras.utils.plot_model(
        default_model,
        to_file="model.png",
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=True,
        dpi=300,
    )

    y_train = default_model.encode_y(base_y_train, n_classes=72, n_hh_classes=12, n_mm_classes=12)
    y_test = default_model.encode_y(base_y_test, n_classes=72, n_hh_classes=12, n_mm_classes=12)


    print(y_train[:5])
    print(y_test[:5])
    #
    # print(len(y_train[0]))
    # print(len(y_test[0]))

    try:
        print("Encoding from hh,mm -> f: ", base_y_train.shape, " -> ", y_train.shape)
    except AttributeError:
        print("Encoding from hh,mm -> f: ", base_y_train.shape, " -> ", len(y_train))

    # import matplotlib
    # matplotlib.use('TkAgg')
    # import matplotlib.pyplot as plt
    # matplotlib.rcParams['figure.dpi'] = 300

    #
    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, constrained_layout=True, figsize=(18, 6))
    # ax1.imshow(x_train[0])
    # ax1.set_title(f"{y_train[0]}")
    # ax2.imshow(x_train[23])
    # ax2.set_title(f"{y_train[23]}")
    # ax3.imshow(x_train[1699])
    # ax3.set_title(f"{y_train[1699]}")
    #
    # plt.show()

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

    # yhat, y = np.linspace(0, 2 * np.pi * 1., 200), np.linspace(0., 2 * np.pi * 1., 200)
    # yyhat, yy = np.meshgrid(yhat, y, sparse=True)
    #
    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, constrained_layout=True, figsize=(18 * 0.65, 6 * 0.6),
    #                                     )
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
    #     y_encoded_sin = np.sin(y)
    #     yhat_encoded_sin = np.sin(yhat)
    #
    #     y_encoded_cos = np.cos(y)
    #     yhat_encoded_cos = np.cos(yhat)
    #
    #     sin_loss = np.square(y_encoded_sin - yhat_encoded_sin)
    #     cos_loss = np.square(y_encoded_cos - yhat_encoded_cos)
    #
    #     loss = sin_loss + cos_loss
    #     return loss, sin_loss, cos_loss
    #
    # loss, sin_loss, cos_loss = sincos(yy, yyhat) #sincos(yy.flatten(), yyhat.flatten())
    #
    # import matplotlib
    # matplotlib.use('TkAgg')
    # import matplotlib.pyplot as plt
    #
    # im1 = ax1.contourf(y, yhat, loss, #.reshape(200, 200),
    #                  ) #vmin=0., vmax=4., extent=[0, 360, 0,  360])
    # ax1.set_title("Loss")
    # ax1.set_xlabel(r"$y$ [rad]")
    # ax1.set_ylabel(r"$\hat{y}$ [rad]")
    # ax1.set_aspect(1)
    #
    # im2 = ax2.contourf(y, yhat, sin_loss, #.reshape(200, 200),
    #                 ) #vmin=0., vmax=4., extent=[0, 360, 0,  360])
    # ax2.set_title("Squared sine loss")
    # ax2.set_xlabel(r"$y$ [rad]")
    # # ax2.set_ylabel(r"$\hat{y}$ [deg]")
    # ax2.set_aspect(1)
    #
    # im3 = ax3.contourf(y, yhat, cos_loss, #.reshape(200, 200),
    #                 ) #vmin=0., vmax=4., extent=[0,360, 0,  360])
    # ax3.set_title("Squared cosine loss")
    # ax3.set_xlabel(r"$y$ [rad]")
    # # ax3.set_ylabel(r"$\hat{y}$ [deg]")
    # ax3.set_aspect(1)
    #
    # plt.colorbar(im1, ax=ax3, label="Loss")
    #
    # plt.show()
    #
    # raise NotImplementedError

    history = default_model.train(x_train, y_train, validation_data=(x_test, y_test),
                                  epochs=10)

    print(history)
    print(history.params)

    default_model.test(x_test, y_test)

    # print(y_test[:5])
    # print(default_model.predict(x_test[:5]))

    # print(tf.argmax(y_test[:5], axis=-1))
    # print(default_model._predict_class(x_test[:5]))

    print("=============")

    # print(y_train[:5])
    # print(default_model.predict(x_train[:5]))

    # print(tf.argmax(y_train[:5], axis=-1))
    # print(default_model._predict_class(x_train[:5]))

    print(base_y_train[:5])
    print(tf.argmax(y_train[:5], axis=-1))
    print(default_model._predict_class(x_train[:5]))

    print(tf.reduce_mean(default_model._class_difference(y_train, default_model.predict(x_train))))
