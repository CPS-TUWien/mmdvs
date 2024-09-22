import tensorflow as tf

class PerImageNormalization(tf.keras.layers.Layer):
    def __init__(self):
        super(PerImageNormalization, self).__init__()

    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float32)
        return tf.vectorized_map(tf.image.per_image_standardization, inputs)

    def compute_output_shape(self, input_shape):
        return input_shape