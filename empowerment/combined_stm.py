import tensorflow as tf
import numpy as np
import os
from PIL import Image

class CombinedStateTransitionModel(tf.keras.Model):
    def __init__(self, input_shape, latent_dim, hidden_layers,
                 hidden_dim, convolution_layers, convolution_size,
                 convolution_filters, stm_layers, stm_dim, action_dim,
                 noise = 0.0, regularizer = None, dropout = 0.0):
        super(CombinedStateTransitionModel, self).__init__()

        self.encoder = self.init_encoder(
            input_shape, latent_dim, hidden_layers,
            hidden_dim, convolution_layers, convolution_size,
            convolution_filters, regularizer, dropout)

        self.decoder = self.init_decoder(
            input_shape, latent_dim, hidden_layers,
            hidden_dim, convolution_layers, convolution_size,
            convolution_filters, regularizer, dropout)

        self.state_transition = self.init_state_transition(
            latent_dim, stm_layers, stm_dim,
            noise, regularizer, dropout)

        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        self.build([(None,) + input_shape, (None, action_dim)])

    def call(self, x):
        '''
        Input: [(batch, img_x, img_y, img_channel), (batch, actions)]
        Output: (batch, timestep, img_x, img_y, img_channel)
        There are always two timesteps, the current and the next.
        '''
        image, action = x
        encoded = self.encoder(image)
        next_encoded = self.state_transition(
            tf.concat([action, encoded], 1))
        return tf.stack([
            self.decoder(encoded),
            self.decoder(next_encoded)
            ], axis = 1)

    def build(self, input_shape):
        image_shape, action_shape = input_shape
        assert(image_shape == (None,) + self.input_shape)
        assert(action_shape == (None, self.action_dim))
        self.encoder.build(image_shape)
        self.decoder.build((None, self.latent_dim))
        self.state_transition.build(
            (None, self.latent_dim + self.action_dim))

    def init_encoder(self, input_shape, latent_dim, hidden_layers,
                     hidden_dim, convolution_layers, convolution_size,
                     convolution_filters, regularizer, dropout):
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
            encoder_stack.append(
                tf.keras.layers.Dense(hidden_dim,
                                      kernel_regularizer = regularizer,
                                      bias_regularizer = regularizer,
                                      activation = tf.nn.relu))
            if dropout > 0.0:
                encoder_stack.append(
                    tf.keras.layers.Dropout(dropout))

        encoder_stack.append(
            tf.keras.layers.Dense(latent_dim,
                                  kernel_regularizer = regularizer,
                                  bias_regularizer = regularizer,
                                  activation = tf.nn.sigmoid))

        return tf.keras.Sequential(
            [tf.keras.layers.InputLayer(shape = input_shape)] +
            convolution_stack +
            [tf.keras.layers.Flatten()] +
            encoder_stack,
            name = 'encoder')

    def init_decoder(self, input_shape, latent_dim, hidden_layers,
                     hidden_dim, convolution_layers, convolution_size,
                     convolution_filters, regularizer, dropout):
        conv_edges = convolution_layers * (convolution_size - 1)
        deconvolution_input_shape = (
            input_shape[0] - conv_edges,
            input_shape[1] - conv_edges,
            convolution_filters if convolution_layers > 0 else input_shape[2])
        dense_output_shape = (deconvolution_input_shape[0] *
                              deconvolution_input_shape[1] *
                              deconvolution_input_shape[2])

        decoder_stack = []
        for i in range(hidden_layers):
            decoder_stack.append(
                tf.keras.layers.Dense(hidden_dim,
                                      kernel_regularizer = regularizer,
                                      bias_regularizer = regularizer,
                                      activation = tf.nn.relu))
            if dropout > 0.0:
                decoder_stack.append(
                    tf.keras.layers.Dropout(dropout))

        decoder_stack.append(
            tf.keras.layers.Dense(dense_output_shape,
                                  kernel_regularizer = regularizer,
                                  bias_regularizer = regularizer,
                                  activation = tf.nn.relu))

        deconvolution_stack = []
        for i in range(convolution_layers - 1):
            if dropout > 0.0:
                deconvolution_stack.append(
                    tf.keras.layers.SpatialDropout2D(dropout))
            deconvolution_stack.append(
                tf.keras.layers.Conv2DTranspose(
                    convolution_filters,
                    convolution_size,
                    kernel_regularizer = regularizer,
                    bias_regularizer = regularizer))
        if convolution_layers >= 1:
            if dropout > 0.0:
                deconvolution_stack.append(
                    tf.keras.layers.SpatialDropout2D(dropout))
            deconvolution_stack.append(
                tf.keras.layers.Conv2DTranspose(
                    input_shape[2],
                    convolution_size,
                    activation = tf.nn.sigmoid,
                    kernel_regularizer = regularizer,
                    bias_regularizer = regularizer))

        return tf.keras.Sequential(
            decoder_stack +
            [tf.keras.layers.Reshape(deconvolution_input_shape)] +
            deconvolution_stack,
            name = 'decoder')

    def init_state_transition(self, latent_dim, stm_layers, stm_dim,
                              noise, regularizer, dropout):
        stm_stack = []
        stm_stack.append(
            tf.keras.layers.GaussianNoise(noise))

        for i in range(stm_layers):
            stm_stack.append(
                tf.keras.layers.Dense(stm_dim,
                                      kernel_regularizer = regularizer,
                                      bias_regularizer = regularizer,
                                      activation = tf.nn.relu))
            if dropout > 0.0:
                stm_stack.append(
                    tf.keras.layers.Dropout(dropout))

        stm_stack.append(
            tf.keras.layers.Dense(latent_dim,
                                  kernel_regularizer = regularizer,
                                  bias_regularizer = regularizer,
                                  activation = tf.nn.sigmoid))

        return tf.keras.Sequential(
            stm_stack,
            name = 'state_transition')

class SaveImageCallback(tf.keras.callbacks.Callback):
    '''A callback to save output images at the end of every epoch.'''
    def __init__(self, model_input, save_dir):
        self.model_input = model_input
        self.save_dir = save_dir

    def on_epoch_end(self, epoch, logs=None):
        # Don't ruin a training run if something goes wrong here.
        try:
            model_output = self.model(self.model_input)
            image = Image.fromarray(
                np.asarray(
                    model_output[0][0] * 255
                ).astype(np.uint8)
            )
            path = os.path.join(self.save_dir, 'epoch_{}_autoencoded.png'.format(epoch))
            image.save(path)

            image = Image.fromarray(
                np.asarray(
                    model_output[0][1] * 255
                ).astype(np.uint8)
            )
            path = os.path.join(self.save_dir, 'epoch_{}_predicted.png'.format(epoch))
            image.save(path)
        except Exception as e:
            print(e)

if __name__ == '__main__':
    input_shape = (8, 8, 1)
    latent_dim = 2
    hidden_layers = 1
    hidden_dim = 4
    convolution_layers = 1
    convolution_size = 3
    convolution_filters = 2
    stm_layers = 2
    stm_dim = 4
    action_dim = 2

    m = CombinedStateTransitionModel(
        input_shape, latent_dim, hidden_layers,
        hidden_dim, convolution_layers, convolution_size,
        convolution_filters, stm_layers, stm_dim, action_dim)

    print(m([tf.eye(8, batch_shape = (2,))[:,:,:,None], tf.eye(2)]))
