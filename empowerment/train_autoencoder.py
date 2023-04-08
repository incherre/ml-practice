import tensorflow as tf
import numpy as np
import os

import autoencoder
import cartpole_generator

model_save_path = os.path.abspath(
    os.path.join('.', 'models', 'autoencoder.weights'))

val_path = os.path.abspath(
    os.path.join('.', 'data', 'rand_trajectories.npy'))
val_data = np.load(val_path) / 255

autoencoder_model = autoencoder.Autoencoder(
    val_data[0].shape, 16, 0, None)
autoencoder_model.summary()
autoencoder_model.compile(
    optimizer = tf.keras.optimizers.Adam(),
    loss = tf.keras.losses.MeanSquaredError())

try:
    autoencoder_model.fit(
        cartpole_generator.CartPoleFrameGen(100),
        epochs=100,
        validation_data=(val_data, val_data))
except Exception as e:
    print(e)

save_model = input('Save model? (y/n)')
if save_model.strip().lower()[0] == 'y':
    print('Saving...')
    autoencoder_model.save_weights(model_save_path)
else:
    print('Not saving.')
