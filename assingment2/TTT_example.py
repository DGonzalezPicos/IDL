from TTTcnn import *

import tensorflow as tf
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Input
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
from dataclasses import dataclass
from sklearn.utils import shuffle
import math
from functools import partial
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, mean_absolute_error


@dataclass
class Dataset():
  name: str
  input_shape: object
  x_train: object
  y_train: object
  x_valid: object
  y_valid: object
  x_test: object
  y_test: object


def decimal_representation_of(y):
  return y[:,0] + y[:,1] / 60

def cyclical_representation_of(y):
  decimal_y = decimal_representation_of(y)
  return np.array([np.sin(2*np.pi*decimal_y/12), np.cos(2*np.pi*decimal_y/12)])

def cyclical_representation_of_hours(hours):
  return np.array([np.sin(2*np.pi*hours/12), np.cos(2*np.pi*hours/12)])

def cyclical_representation_of_minutes(minutes):
  return np.array([np.sin(2*np.pi*minutes/60), np.cos(2*np.pi*minutes/60)])


def represented_in_range(number, interval_in_minutes):
  return [
          1 if number >= i and number < i + interval_in_minutes/60 else 0
          for i in np.arange(0, 12, interval_in_minutes / 60)
          ]

def grouped_in_classes(y, interval_in_minutes=30):
  decimal_y = decimal_representation_of(y)
  return np.array(
      [
       represented_in_range(yi, interval_in_minutes)
       for yi in decimal_y
      ])


class DecimalTimesMeanLoss(Loss):

    def call(self, y_true, y_pred):
        return tf.reduce_mean(
            tf.math.minimum(
                tf.math.abs(y_true - y_pred),
                tf.math.abs(tf.math.minimum(y_true, y_pred) + 12 - tf.math.maximum(y_true, y_pred))
            )
        )


class MinutesMeanLoss(Loss):

    def call(self, y_true, y_pred):
        return tf.reduce_mean(
            tf.math.minimum(
                tf.math.abs(y_true - y_pred),
                tf.math.abs(tf.math.minimum(y_true, y_pred) + 60 - tf.math.maximum(y_true, y_pred))
            )
        )


class CyclicalTimesDistanceMeanLoss(Loss):

    def call(self, y_true, y_pred):
        loss = tf.reduce_mean(
            tf.math.sqrt(tf.reduce_sum(
                tf.math.square(y_true - y_pred),
                axis=1
            ))
        )
        return loss


class CyclicalTimesMinutesMeanLoss(Loss):

    def call(self, y_true, y_pred):
        dot_product = tf.reduce_sum(tf.multiply(y_pred, y_true), axis=1)
        y_pred_norm = tf.norm(y_pred, axis=1)
        y_true_norm = tf.norm(y_true, axis=1)
        multiplied_norms = tf.multiply(y_pred_norm, y_true_norm)

        arccos = tf.math.acos(dot_product / multiplied_norms)
        arccos = tf.where(tf.math.is_nan(arccos), tf.zeros_like(arccos), arccos)

        return 60 * (arccos / (2 * np.pi))


class DecimalTimesCosineSimilarityLoss(Loss):

    def call(self, y_true, y_pred):
        y_cyclical_true = tf.map_fn(lambda x: [tf.math.sin(2 * np.pi * x / 12), tf.math.cos(2 * np.pi * x / 12)],
                                    y_true, dtype=[tf.float32, tf.float32])
        y_cyclical_pred = tf.map_fn(lambda x: [tf.math.sin(2 * np.pi * x / 12), tf.math.cos(2 * np.pi * x / 12)],
                                    y_pred, dtype=[tf.float32, tf.float32])

        cosine_loss = CosineSimilarity(axis=1)
        return cosine_loss(y_cyclical_true, y_cyclical_pred)


def adjusted_mae_numpy(a, b, max_value):
    return np.average(np.min(np.concatenate((np.abs(a - b), np.abs(
        np.min(np.concatenate((a, b), axis=1), axis=1) + max_value - np.max(np.concatenate((a, b), axis=1),
                                                                            axis=1)).reshape(-1, 1)), axis=1), axis=1))


def mean_minutes_loss_for_cyclical_time(y_true, y_pred):
    y_pred_unit_vectors = tf.map_fn(lambda x: x / tf.norm(x), y_pred)
    y_true_unit_vectors = tf.map_fn(lambda x: x / tf.norm(x), y_true)
    print(y_pred_unit_vectors)
    print(y_true_unit_vectors)
    minutes_losses = tf.map_fn(
        lambda i:
        60 * 12 * tf.acos(tf.tensordot(y_pred_unit_vectors[i], y_true_unit_vectors[i], 1)) / (2 * tf.constant(np.pi)),
        tf.range(y_pred_unit_vectors.shape[0])
    )
    return tf.reduce_mean(minutes_losses)


def mean_minutes_loss_metric(y_true, y_pred):
    return tf.reduce_mean(
        tf.math.minimum(
            tf.math.abs(y_true - y_pred),
            tf.math.abs(tf.math.minimum(y_true, y_pred) + 12 - tf.math.maximum(y_true, y_pred))
        )
    ) * 60


def regression_cnn_1(loss, dataset, output_units, metric=None):
    DefaultConv2D = partial(Conv2D, kernel_size=3, activation='leaky_relu', padding="VALID")

    model = keras.models.Sequential([
        Rescaling(1. / 255, input_shape=dataset.input_shape),
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
        Dense(units=output_units, activation='linear'),
    ])

    if metric != None:
        model.compile(optimizer='adam', loss=loss, metrics=[metric])
    else:
        model.compile(optimizer='adam', loss=loss)

    model.summary()
    return model


def classification_cnn(dataset, classes, loss='categorical_crossentropy'):
    tf.random.set_seed(42)
    DefaultConv2D = partial(Conv2D, kernel_size=3, activation='leaky_relu', kernel_initializer='he_normal')

    model = keras.models.Sequential([
        Rescaling(1. / 255, input_shape=dataset.input_shape),
        DefaultConv2D(filters=16, kernel_size=5),
        MaxPooling2D(pool_size=(2, 2)),
        DefaultConv2D(filters=32),
        DefaultConv2D(filters=32),
        MaxPooling2D(pool_size=(2, 2)),
        DefaultConv2D(filters=64),
        DefaultConv2D(filters=64),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.5),
        Flatten(),
        Dense(units=64, activation='leaky_relu', kernel_initializer='he_normal'),
        Dense(units=64, activation='leaky_relu', kernel_initializer='he_normal'),
        # Dense(units=1024, activation='relu'),
        # Dropout(0.2),
        # Dense(units=512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        # Dense(units=512, activation='relu'),
        # Dropout(0.2),
        Dense(units=classes, activation='softmax'),
    ])

    model.compile(
        optimizer='adam',
        loss=loss,
        metrics=['accuracy'],
    )

    model.summary()
    return model

def train(model, dataset):
  model.fit(
    dataset.x_train,
    dataset.y_train,
    epochs=5,
    batch_size=128,
    validation_data=(dataset.x_valid, dataset.y_valid),
    callbacks=[keras.callbacks.EarlyStopping(patience=10)],
  )

def evaluate(model, x, y):
  score = model.evaluate(x, y, verbose=0)
  return score


def get_angle(sin, cos):
    angle = math.atan2(sin, cos) * 180 / math.pi  # ALWAYS USE THIS

    if angle < 0:
        angle += 360

    return angle


def get_minutes(sin, cos, max_value):
    return int(get_angle(sin, cos) * max_value * 1.0 / 180)


if __name__ == '__main__':
    print("No longer supported, use the supplied jupyter notebook.")
    x_train, base_y_train, x_test, base_y_test = get_data()
    default_model = TellTheTimeCNN()
    input_shape = (150, 150, 1)

    y_train, y_test = default_model.encode_y(base_y_train), default_model.encode_y(base_y_test)

    try:
        print("Encoding from hh,mm -> f: ", base_y_train.shape, " -> ", y_train.shape)
    except AttributeError:
        print("Encoding from hh,mm -> f: ", base_y_train.shape, " -> ", len(y_train))

    n_train = int(len(y_train) * 0.1)

    x_valid, y_valid = x_train[n_train:], y_train[n_train:]
    x_train, y_train = x_train[:n_train], y_train[:n_train]

    print(x_train[:6])
    print(y_train[:6])

    dataset = Dataset(
          name='decimal-representation',
          input_shape=(150, 150, 1),
          x_train=x_train,
          y_train=y_train,
          x_valid=x_valid,
          y_valid=y_valid,
          x_test=x_test,
          y_test=y_test,
      )

    decimal_hours_minutes_custom_loss_model = regression_cnn_1(DecimalTimesMeanLoss(), dataset, 1,
                                                               mean_minutes_loss_metric)
    train(decimal_hours_minutes_custom_loss_model, dataset)
    print(
        f'Minutes loss on test set: {evaluate(decimal_hours_minutes_custom_loss_model, dataset.x_test, dataset.y_test)[1]}')

    print(y_test[:5])
    print(decimal_hours_minutes_custom_loss_model.predict(x_test[:5]))