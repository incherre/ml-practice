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
    def __init__(self, input_shape, latent_dim, hidden_layers,
                 hidden_dim, convolution_layers, convolution_size,
                 convolution_filters, regularizer = None, dropout = 0.0):
        super(Autoencoder, self).__init__()

        convolution_stack = []
        for i in range(convolution_layers):
            if dropout > 0.0:
                convolution_stack.append(
                    tf.keras.layers.SpatialDropout2D(dropout))
            convolution_stack.append(
                tf.keras.layers.Conv2D(convolution_filters,
                                       convolution_size,
                                       kernel_regularizer = regularizer,
                                       bias_regularizer = regularizer))

        encoder_stack = []
        for i in range(hidden_layers):
            encoder_stack.append(DenseForward(hidden_dim,
                                              regularizer = regularizer,
                                              activation_function = tf.nn.relu))
        encoder_stack.append(DenseForward(latent_dim,
                                          regularizer = regularizer,
                                          activation_function = tf.nn.sigmoid))

        dropout_stack = [tf.keras.layers.Dropout(dropout) for i in range(len(encoder_stack))] if dropout > 0.0 else []
        self.encoder = tf.keras.Sequential(
            [tf.keras.layers.InputLayer(input_shape = input_shape)] +
            convolution_stack +
            [tf.keras.layers.Flatten()] +
            [layer for layers in zip(dropout_stack, encoder_stack) for layer in layers] if dropout > 0.0 else encoder_stack,
            name = 'encoder')

        decoder_stack = []
        for enc_layer in reversed(encoder_stack[1:]):
            decoder_stack.append(DenseTied(enc_layer,
                                           activation_function = tf.nn.relu))
            if dropout > 0.0:
                decoder_stack.append(tf.keras.layers.Dropout(dropout))
        decoder_stack.append(DenseTied(
            encoder_stack[0],
            activation_function = tf.nn.relu))

        conv_edges = convolution_layers * (convolution_size - 1)
        deconvolution_input_shape = (input_shape[0] - conv_edges,
                                     input_shape[1] - conv_edges,
                                     convolution_filters if convolution_layers > 0 else input_shape[2])

        deconvolution_stack = []
        for i in range(convolution_layers - 1):
            if dropout > 0.0:
                deconvolution_stack.append(
                    tf.keras.layers.SpatialDropout2D(dropout))
            deconvolution_stack.append(
                tf.keras.layers.Conv2DTranspose(convolution_filters,
                                                convolution_size,
                                                kernel_regularizer = regularizer,
                                                bias_regularizer = regularizer))
        if convolution_layers >= 1:
            if dropout > 0.0:
                deconvolution_stack.append(
                    tf.keras.layers.SpatialDropout2D(dropout))
            deconvolution_stack.append(
                tf.keras.layers.Conv2DTranspose(input_shape[2],
                                                convolution_size,
                                                activation = tf.nn.sigmoid,
                                                kernel_regularizer = regularizer,
                                                bias_regularizer = regularizer))

        self.decoder = tf.keras.Sequential(
            decoder_stack +
            [tf.keras.layers.Reshape(deconvolution_input_shape)] +
            deconvolution_stack,
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
