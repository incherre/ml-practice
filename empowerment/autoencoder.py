import tensorflow as tf
import numpy as np
import os
from PIL import Image

class DenseForward(tf.keras.layers.Layer):
    def __init__(self, num_outputs,
                 activation_function = tf.nn.relu,
                 regularizer = None):
        super(DenseForward, self).__init__()
        self.num_outputs = num_outputs
        self.activation_function = activation_function
        self.regularizer = regularizer
        self.biases = self.add_weight(
            "biases",
            shape=[1, self.num_outputs],
            regularizer = self.regularizer)

    def build(self, input_shape):
        self.nn_weights = self.add_weight(
            "weights",
            shape=(int(input_shape[-1]), self.num_outputs),
            regularizer = self.regularizer)

    def call(self, inputs):
        return self.activation_function(
            tf.matmul(
                inputs,
                self.nn_weights
            ) + self.biases
        )

class DenseTied(tf.keras.layers.Layer):
    def __init__(self, tied_forward_layer, activation_function = None):
        super(DenseTied, self).__init__()
        self.tied_forward_layer = tied_forward_layer
        if not activation_function is None:
            self.activation_function = activation_function
        else:
            self.activation_function = tied_forward_layer.activation_function

    def build(self, input_shape):
        # This is in build because it depends on the source layer
        # being built first.
        self.biases = self.add_weight(
            "biases",
            shape=[1, self.tied_forward_layer.nn_weights.shape[0]],
            regularizer = self.tied_forward_layer.regularizer)

    def call(self, inputs):
        return self.activation_function(
            tf.matmul(
                inputs,
                self.tied_forward_layer.nn_weights,
                transpose_b = True
            ) + self.biases
        )

class Autoencoder(tf.keras.Model):
    def __init__(self, input_shape, latent_dim,
                 hidden_layers, hidden_dim, regularizer = None, dropout = 0.0):
        super(Autoencoder, self).__init__()

        encoder_stack = []
        for i in range(hidden_layers):
            encoder_stack.append(DenseForward(hidden_dim,
                                              regularizer = regularizer))
        encoder_stack.append(DenseForward(latent_dim,
                                          regularizer = regularizer))

        dropout_stack = [tf.keras.layers.Dropout(dropout) for i in range(len(encoder_stack))]
        self.encoder = tf.keras.Sequential(
            [tf.keras.layers.Flatten()] +
            [layer for layers in zip(dropout_stack, encoder_stack) for layer in layers],
            name = 'encoder')

        decoder_stack = []
        for enc_layer in reversed(encoder_stack[1:]):
            decoder_stack.append(DenseTied(enc_layer))
        decoder_stack.append(DenseTied(
            encoder_stack[0],
            activation_function = tf.nn.sigmoid))

        dropout_stack = [tf.keras.layers.Dropout(dropout) for i in range(len(decoder_stack))]
        self.decoder = tf.keras.Sequential(
            [layer for layers in zip(dropout_stack, decoder_stack) for layer in layers] +
            [tf.keras.layers.Reshape(input_shape)],
            name = 'decoder')

        self.build((None,) + input_shape)

    def call(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)

class SaveImageCallback(tf.keras.callbacks.Callback):
    '''A callback to save autoencoded images at the end of every epoch.'''
    def __init__(self, image_sample, save_dir):
        self.image_sample = image_sample
        self.save_dir = save_dir

    def on_epoch_end(self, epoch, logs=None):
        # Don't ruin a training run if something goes wrong here.
        try:
            image = Image.fromarray(
                np.asarray(
                    self.model(self.image_sample[None, :, :, :])[0] * 255
                ).astype(np.uint8)
            )
            path = os.path.join(self.save_dir, 'epoch_{}.png'.format(epoch))
            image.save(path)
        except Exception as e:
            print(e)
