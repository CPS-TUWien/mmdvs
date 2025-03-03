import tensorflow as tf
from tensorflow.keras.models import Model
from image_normalization import PerImageNormalization
import numpy as np


def setup_model(input_dim, output_dim, rnn = None, stateful = False):
    W, H, C = input_dim
    if stateful:
        input = tf.keras.layers.Input(batch_shape=(1, 1, H, W, C), name="inputs")
    else:
        input = tf.keras.layers.Input(shape=(None, H, W, C), name="inputs")

    num_neuron_conv = 64

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
            tf.keras.layers.Dense(num_neuron_conv, name="dense"),
        ]
    )

   
    td_conv = tf.keras.layers.TimeDistributed(conv_head)(input)

    # Input mapping
    td_conv = tf.keras.layers.Dense(num_neuron_conv, name="dense_1")(td_conv)

    # Passing through the RNN
    td_conv = rnn(td_conv)

    # Output mapping
    y = tf.keras.layers.Dense(output_dim, name="dense_2", kernel_initializer="ones")(td_conv)

    model = Model(input, y)

    return model