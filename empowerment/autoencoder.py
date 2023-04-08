import tensorflow as tf

class DenseForward(tf.keras.layers.Layer):
    def __init__(self, num_outputs, activation_function = tf.nn.relu):
        super(DenseForward, self).__init__()
        self.num_outputs = num_outputs
        self.activation_function = activation_function
        self.biases = self.add_weight(
            "biases",
            shape=[1, self.num_outputs])

    def build(self, input_shape):
        self.nn_weights = self.add_weight(
            "weights",
            shape=(int(input_shape[-1]), self.num_outputs))

    def call(self, inputs):
        return self.activation_function(
            tf.matmul(
                inputs,
                self.nn_weights
            ) + self.biases
        )

class DenseTied(tf.keras.layers.Layer):
    def __init__(self, tied_forward_layer):
        super(DenseTied, self).__init__()
        self.tied_forward_layer = tied_forward_layer

    def build(self, input_shape):
        # This is in build because it depends on the source layer
        # being built first.
        self.biases = self.add_weight(
            "biases",
            shape=[1, self.tied_forward_layer.nn_weights.shape[0]])

    def call(self, inputs):
        return self.tied_forward_layer.activation_function(
            tf.matmul(
                inputs,
                self.tied_forward_layer.nn_weights,
                transpose_b = True
            ) + self.biases
        )

class Autoencoder(tf.keras.Model):
    def __init__(self, input_shape, latent_dim,
                 hidden_layers, hidden_dim):
        super(Autoencoder, self).__init__()

        encoder_stack = []
        for i in range(hidden_layers):
            encoder_stack.append(DenseForward(hidden_dim))
        encoder_stack.append(DenseForward(latent_dim))

        self.encoder = tf.keras.Sequential(
            [tf.keras.layers.Flatten()] + encoder_stack)

        decoder_stack = []
        for enc_layer in reversed(encoder_stack):
            decoder_stack.append(DenseTied(enc_layer))

        self.decoder = tf.keras.Sequential(
            decoder_stack + [tf.keras.layers.Reshape(input_shape)])

        self.build((None,) + input_shape)

    def call(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)
