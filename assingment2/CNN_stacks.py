from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Input, SpatialDropout2D, \
    LocallyConnected2D, LayerNormalization, BatchNormalization, Concatenate
from tensorflow.keras.layers.experimental.preprocessing import Rescaling

def get_CNN_default_block(filters=64, kernel_size=3, hidden_actfn="relu", kernel_initializer="he_uniform"):
    stack = [
        Conv2D(filters=filters, kernel_size=kernel_size, strides=1,
               activation=hidden_actfn, kernel_initializer=kernel_initializer),
        BatchNormalization(axis=1),
        MaxPooling2D(pool_size=2),
    ]
    return stack

def get_CNN_encode(hidden_actfn="relu", kernel_initializer="he_uniform"):
    main_stack = [
        Rescaling(1. / 255., input_shape=(150, 150, 1)),

        Conv2D(filters=64, kernel_size=3,
               activation=hidden_actfn, kernel_initializer=kernel_initializer),
        Conv2D(filters=64, kernel_size=3,
               activation=hidden_actfn, kernel_initializer=kernel_initializer),
        BatchNormalization(axis=1),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(filters=32, kernel_size=3,
               activation=hidden_actfn, kernel_initializer=kernel_initializer),
        Conv2D(filters=32, kernel_size=3,
               activation=hidden_actfn, kernel_initializer=kernel_initializer),
        BatchNormalization(axis=1),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(filters=16, kernel_size=3,
               activation=hidden_actfn, kernel_initializer=kernel_initializer),
        BatchNormalization(axis=1),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
    ]
    return main_stack

def get_CNN_decode(hidden_actfn="relu", kernel_initializer="he_uniform"):
    main_stack = [
        Rescaling(1. / 255., input_shape=(150, 150, 1)),

        Conv2D(filters=16, kernel_size=3,
               activation=hidden_actfn, kernel_initializer=kernel_initializer),
        BatchNormalization(axis=1),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(filters=32, kernel_size=3,
               activation=hidden_actfn, kernel_initializer=kernel_initializer),
        Conv2D(filters=32, kernel_size=3,
               activation=hidden_actfn, kernel_initializer=kernel_initializer),
        BatchNormalization(axis=1),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(filters=64, kernel_size=3,
               activation=hidden_actfn, kernel_initializer=kernel_initializer),
        Conv2D(filters=64, kernel_size=3,
               activation=hidden_actfn, kernel_initializer=kernel_initializer),
        BatchNormalization(axis=1),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
    ]
    return main_stack

def get_DCNN_stack(hidden_actfn="relu", kernel_initializer="he_normal"):
    """
    DCNN with encoding and decoding, but without skip connections. Relatively shallow.
    :param hidden_actfn:
    :param kernel_initializer:
    :return:
    """
    main_stack = [
        Rescaling(1. / 255., input_shape=(150, 150, 1)),

        Conv2D(filters=64, kernel_size=3, strides=1,
               activation=hidden_actfn, kernel_initializer=kernel_initializer, padding="VALID"),
        Conv2D(filters=64, kernel_size=3, strides=1,
               activation=hidden_actfn, kernel_initializer=kernel_initializer, padding="VALID"),
        BatchNormalization(axis=1),
        MaxPooling2D(pool_size=2),

        Conv2D(filters=32, kernel_size=3, strides=1,
               activation=hidden_actfn, kernel_initializer=kernel_initializer, padding="VALID"),
        Conv2D(filters=32, kernel_size=3, strides=1,
               activation=hidden_actfn, kernel_initializer=kernel_initializer, padding="VALID"),
        BatchNormalization(axis=1),
        MaxPooling2D(pool_size=2),

        Conv2D(filters=16, kernel_size=5, strides=1,
               activation=hidden_actfn, kernel_initializer=kernel_initializer, padding="VALID"),
        BatchNormalization(axis=1),
        MaxPooling2D(pool_size=2, strides=1),

        Conv2D(filters=32, kernel_size=3, strides=1,
               activation=hidden_actfn, kernel_initializer=kernel_initializer, padding="VALID"),
        Conv2D(filters=32, kernel_size=3, strides=1,
               activation=hidden_actfn, kernel_initializer=kernel_initializer, padding="VALID"),
        BatchNormalization(axis=1),
        MaxPooling2D(pool_size=2),

        Conv2D(filters=64, kernel_size=3, strides=1,
               activation=hidden_actfn, kernel_initializer=kernel_initializer, padding="VALID"),
        Conv2D(filters=64, kernel_size=3, strides=1,
               activation=hidden_actfn, kernel_initializer=kernel_initializer, padding="VALID"),
        BatchNormalization(axis=1),
        MaxPooling2D(pool_size=2),

        Flatten(),
    ]
    return main_stack

def get_simple_classification_CNN(hidden_actfn="relu", kernel_initializer="he_normal"):
    main_stack = [
        Rescaling(1. / 255., input_shape=(150, 150, 1)),

        Conv2D(filters=16, kernel_size=5,
               activation=hidden_actfn, kernel_initializer=kernel_initializer),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(filters=32, kernel_size=3,
               activation=hidden_actfn, kernel_initializer=kernel_initializer),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(filters=64, kernel_size=3,
               activation=hidden_actfn, kernel_initializer=kernel_initializer),
        MaxPooling2D(pool_size=(2, 2)),

        Dropout(.5),
        Flatten(),
    ]
    return main_stack