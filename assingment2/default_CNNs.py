from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Input, SpatialDropout2D, \
    LocallyConnected2D, LayerNormalization, BatchNormalization, Concatenate
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from CNN_stacks import *
from TTTcnn import *


def single_head_classification(n_classes=72, hidden_actfn="leaky_relu", kernel_initializer="he_normal", **kwargs):
    stack = [
        Rescaling(1. / 255., input_shape=(150, 150, 1)),

        Conv2D(filters=64, kernel_size=5,
               activation=hidden_actfn, kernel_initializer=kernel_initializer),
        MaxPooling2D(pool_size=(2, 2), strides=2),
        BatchNormalization(),

        Conv2D(filters=128, kernel_size=3,
               activation=hidden_actfn, kernel_initializer=kernel_initializer),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),

        Conv2D(filters=256, kernel_size=3,
               activation=hidden_actfn, kernel_initializer=kernel_initializer),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),

        Conv2D(filters=512, kernel_size=3,
               activation=hidden_actfn, kernel_initializer=kernel_initializer),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),

        Conv2D(filters=1024, kernel_size=3,
               activation=hidden_actfn, kernel_initializer=kernel_initializer),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),

        Flatten(),
    ]
    settings = {
        "learning_rate": 1.0e-3,
        "encoding": "common_classes",
        "type": ["classification"],  # classification, regression (can be sequence)
        "actfn_normalization": ["softmax"],  # must be sequence if type is sequence
        "loss": ["categorical"],  # must be sequence if type is sequence
        "n_classes": [n_classes],  # must be sequence if type is sequence
        "decay": 5.0e-3,
        "scheduler": True,
        "main_stack": stack,
        "metric": "classes"
    }
    return settings

def single_large_head_regression(hidden_actfn="leaky_relu", kernel_initializer="he_normal", **kwargs):
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

        Flatten(),
    ]
    settings = {
        "learning_rate": 1.0e-3,
        "encoding": "large_decimal",
        "type": ["regression"],  # classification, regression (can be sequence)
        "actfn_normalization": ["linear"],  # must be sequence if type is sequence
        "loss": ["mse"],  # must be sequence if type is sequence
        "n_classes": [0],  # must be sequence if type is sequence
        "decay": 5.0e-5,
        "scheduler": True,
        "main_stack": main_stack
    }
    return settings

def single_head_regression(hidden_actfn="leaky_relu", kernel_initializer="he_normal", **kwargs):
    main_stack = [
        Rescaling(1. / 255., input_shape=(150, 150, 1)),

        Conv2D(filters=32, kernel_size=5,
               activation=hidden_actfn, kernel_initializer=kernel_initializer),
        MaxPooling2D(pool_size=(2, 2), strides=1),
        BatchNormalization(),

        Conv2D(filters=64, kernel_size=3,
               activation=hidden_actfn, kernel_initializer=kernel_initializer),
        MaxPooling2D(pool_size=(2, 2), strides=1),
        BatchNormalization(),

        Conv2D(filters=128, kernel_size=3,
               activation=hidden_actfn, kernel_initializer=kernel_initializer),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),

        Conv2D(filters=256, kernel_size=3,
               activation=hidden_actfn, kernel_initializer=kernel_initializer),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),

        Conv2D(filters=512, kernel_size=3,
               activation=hidden_actfn, kernel_initializer=kernel_initializer),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),

        Flatten(),
    ]
    settings = {
        "learning_rate": 1.0e-2,
        "encoding": "decimal",
        "type": ["regression"],  # classification, regression (can be sequence)
        "actfn_normalization": ["linear"],  # must be sequence if type is sequence
        "loss": ["linear_decimal_cyclic"],  # must be sequence if type is sequence
        "n_classes": [0],  # must be sequence if type is sequence
        "decay": 5.0e-5,
        "scheduler": False,
        "main_stack": main_stack
    }
    return settings

def double_head_regression(hidden_actfn="leaky_relu", kernel_initializer="he_normal", **kwargs):
    main_stack = [
        Rescaling(1. / 255., input_shape=(150, 150, 1)),

        # Conv2D(filters=16, kernel_size=5,
        #        activation=hidden_actfn, kernel_initializer=kernel_initializer),
        # MaxPooling2D(pool_size=(2, 2)),
        #
        Conv2D(filters=32, kernel_size=3,
               activation=hidden_actfn, kernel_initializer=kernel_initializer),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(filters=64, kernel_size=3,
               activation=hidden_actfn, kernel_initializer=kernel_initializer),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(filters=128, kernel_size=3,
               activation=hidden_actfn, kernel_initializer=kernel_initializer),
        MaxPooling2D(pool_size=(2, 2)),

        # Conv2D(filters=256, kernel_size=3,
        #        activation=hidden_actfn, kernel_initializer=kernel_initializer),
        # MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
    ]
    settings = {
        "learning_rate": 1.0e-2,
        "encoding": "sin_cos",
        "type": ["regression", "regression"],  # classification, regression (can be sequence)
        "actfn_normalization": ["linear", "linear"],  # must be sequence if type is sequence
        "loss": ["2out_regression", "2out_regression"],  # must be sequence if type is sequence
        "n_classes": [0, 0],  # must be sequence if type is sequence
        "decay": 1.0e-4,
        "scheduler": True,
        "main_stack": main_stack,
        "n_outputs": [2, 2]
    }
    return settings

def big_double_head_regression(hidden_actfn="leaky_relu", kernel_initializer="he_normal", **kwargs):
    main_stack = [
        Rescaling(1. / 255., input_shape=(150, 150, 1)),

        Conv2D(filters=64, kernel_size=5,
               activation=hidden_actfn, kernel_initializer=kernel_initializer),
        MaxPooling2D(pool_size=(2, 2), strides=2),
        BatchNormalization(),

        Conv2D(filters=128, kernel_size=3,
               activation=hidden_actfn, kernel_initializer=kernel_initializer),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),

        Conv2D(filters=256, kernel_size=3,
               activation=hidden_actfn, kernel_initializer=kernel_initializer),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),

        Conv2D(filters=512, kernel_size=3,
               activation=hidden_actfn, kernel_initializer=kernel_initializer),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),

        Conv2D(filters=1024, kernel_size=3,
               activation=hidden_actfn, kernel_initializer=kernel_initializer),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),

        # Conv2D(filters=2048, kernel_size=3,
        #        activation=hidden_actfn, kernel_initializer=kernel_initializer),
        # MaxPooling2D(pool_size=(2, 2)),
        # BatchNormalization(),

        Flatten(),
    ]
    settings = {
        "learning_rate": 1.0e-2,
        "encoding": "sin_cos",
        "type": ["regression", "regression"],  # classification, regression (can be sequence)
        "actfn_normalization": ["tanh", "tanh"],  # must be sequence if type is sequence
        "loss": ["2out_regression", "2out_regression"],  # must be sequence if type is sequence
        "n_classes": [0, 0],  # must be sequence if type is sequence
        "decay": 1.0e-6,
        "scheduler": False,
        "main_stack": main_stack,
        "n_outputs": [2, 2],
        "metric": "2head"
    }
    return settings

def quad_head_regression(hidden_actfn="leaky_relu", kernel_initializer="he_normal", **kwargs):
    main_stack = [
        Rescaling(1. / 255., input_shape=(150, 150, 1)),

        Conv2D(filters=128, kernel_size=3,
               activation=hidden_actfn, kernel_initializer=kernel_initializer),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(filters=256, kernel_size=3,
               activation=hidden_actfn, kernel_initializer=kernel_initializer),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(filters=512, kernel_size=3,
               activation=hidden_actfn, kernel_initializer=kernel_initializer),
        MaxPooling2D(pool_size=(2, 2)),

        # Conv2D(filters=256, kernel_size=3,
        #        activation=hidden_actfn, kernel_initializer=kernel_initializer),
        # MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
    ]
    settings = {
        "learning_rate": 1.0e-2,
        "encoding": "sin_cos",
        "type": ["regression", "regression", "regression", "regression"],  # classification, regression (can be sequence)
        "actfn_normalization": ["linear", "linear", "linear", "linear"],  # must be sequence if type is sequence
        "loss": ["mse", "mse", "mse", "mse"],  # must be sequence if type is sequence
        "n_classes": [0, 0, 0, 0],  # must be sequence if type is sequence
        "decay": 1.0e-4,
        "scheduler": True,
        "main_stack": main_stack
    }
    return settings

if __name__ == '__main__':
    x_train, base_y_train, x_test, base_y_test = get_data()

    settings = big_double_head_regression()

    default_model = TellTheTimeCNN(settings=settings)

    # tf.keras.utils.plot_model(
    #     default_model,
    #     to_file="model.png",
    #     show_shapes=True,
    #     show_dtype=True,
    #     show_layer_names=True,
    #     rankdir="TB",
    #     expand_nested=True,
    #     dpi=300,
    # )

    y_train = default_model.encode_y(base_y_train)
    y_test = default_model.encode_y(base_y_test)

    try:
        print("Encoding from hh,mm -> f: ", base_y_train.shape, " -> ", y_train.shape)
    except AttributeError:
        print("Encoding from hh,mm -> f: ", base_y_train.shape, " -> ", len(y_train))


    history = default_model.train(x_train, y_train, validation_data=(x_test, y_test),
                                  epochs=25)
    test = default_model.test(x_test, y_test)

    print(history.params)
    print(history.history)

    print(test)

    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['figure.dpi'] = 200

    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 6))

    ax1.set_title(f"Loss")
    ax1.plot(12. * 60. * np.array(history.history["loss"]), label="Training")
    ax1.plot(12. * 60. * np.array(history.history["val_loss"]), label="Validation")
    ax1.scatter(len(history.history["loss"]) - 1, 12. * 60. * test['loss'], label="Test", c="black")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Error/Loss [minutes]")
    ax1.legend()

    # ax2.set_title(f"Accuracy")
    # ax2.plot(history.history["accuracy"], label="Training")
    # ax2.plot(history.history["val_accuracy"], label="Validation")
    # ax2.scatter(len(history.history["accuracy"]) - 1, test['accuracy'], label="Test", c="black")
    # ax2.set_xlabel("Epoch")
    # ax2.set_ylabel("Accuracy")
    # ax2.legend()

    plt.show()



    print(y_train[:5])
    train_pred = default_model.predict(x_train[:5])
    print(train_pred)

    # print(y_train[:5] - train_pred)

    print("===================")

    print(base_y_train[:5])
    print(default_model._predict_4head_reg_labels(x_train[:5]))