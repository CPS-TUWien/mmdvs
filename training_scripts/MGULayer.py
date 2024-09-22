import tensorflow as tf

class MGU(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(MGU, self).__init__(**kwargs)

    def build(self, input_shape):        
        self.forget_kernel = self.add_weight(
            shape=(input_shape[-1] + self.units, self.units),
            initializer="glorot_uniform",
            name="forget_kernel",
        )
        
        self.hidden_kernel = self.add_weight(
            shape=(input_shape[-1] + self.units, self.units),
            initializer="glorot_uniform",
            name="hidden_kernel",
        )
        
        self.forget_bias = self.add_weight(
            shape=(self.units),
            initializer=tf.keras.initializers.Zeros(),
            name="forget_bias",
        )
        
        self.hidden_bias = self.add_weight(
            shape=(self.units),
            initializer=tf.keras.initializers.Zeros(),
            name="hidden_bias",
        )
        
        self.built = True

    def call(self, inputs, states):         
        h_prev = states[0]
        
        fused_input = tf.concat([inputs, h_prev], axis=-1)
        
        fg = tf.nn.sigmoid(tf.matmul(fused_input, self.forget_kernel) + self.forget_bias)
        
        h_tilde = tf.nn.tanh(tf.matmul(tf.concat([inputs, fg * h_prev], axis=-1), self.hidden_kernel) + self.hidden_bias)
        
        ht = (1 - fg) * h_prev + fg * h_tilde
        return ht, [ht]