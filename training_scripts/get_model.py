import tensorflow as tf
from tensorflow.python.keras.models import Model
from image_normalization import PerImageNormalization
from wire_neurons import WiredNeurons
import numpy as np


def store_weights(model, filename):
    """
    Stores weights of model in a Numpy file. This function is needed instead of the built-in weight saving
    methods because layer names may be different with with TimeDistributed and non-TimeDistributed version of the model
    """
    serial = {}
    for v in model.variables:
        name = v.name
        # Remove "rnn/" from start
        if name.startswith("rnn/"):
            name = name[len("rnn/") :]
        if name in serial.keys():
            raise ValueError(f"Duplicate weight name: {name}")
        serial[name] = v.numpy()
    np.savez(filename, **serial)


def setup_model(input_dim, output_dim, rnn = None, return_state = False, stateful = False):
    W, H, C = input_dim
    if stateful:
        input = tf.keras.layers.Input(batch_shape=(1, 1, H, W, C), name="inputs")
    else:
        input = tf.keras.layers.Input(shape=(None, H, W, C), name="inputs")

    conv_head = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input(shape=(H, W, C), name="inputs_1"),
            PerImageNormalization(),
            tf.keras.layers.Conv2D(24, (5, 5), strides=(2, 2), activation="relu", padding="same",name="conv2d"),
            tf.keras.layers.Conv2D(36, (5, 5), activation="relu", padding="same",name="conv2d_1"),
            tf.keras.layers.MaxPool2D(name="max_pooling2d_m"),
            tf.keras.layers.Conv2D(48, (3, 3), activation="relu", padding="same",name="conv2d_2"),
            tf.keras.layers.MaxPool2D(name="max_pooling2d_1"),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same",name="conv2d_3"),
            tf.keras.layers.MaxPool2D(name="max_pooling2d_2"),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same",name="conv2d_4"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, name="dense"),
        ]
    )
    
    td_conv = tf.keras.layers.TimeDistributed(conv_head)(input)

    if isinstance(rnn, WiredNeurons):
        y = rnn(td_conv)
    else:
        # Input mapping (this is included in ltcs)
        td_conv = tf.keras.layers.Dense(64, name="dense_1")(td_conv)

        # Skip this step for pure CNN model
        if rnn is not None:
            td_conv = rnn(td_conv)

        # Output mapping
        y = tf.keras.layers.Dense(output_dim, name="dense_2")(td_conv)

    model = Model(input, y)

    return model

