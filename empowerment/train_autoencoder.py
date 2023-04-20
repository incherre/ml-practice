import tensorflow as tf
import numpy as np
import keras_tuner as kt
import os

import autoencoder
import cartpole_generator

model_save_path = os.path.abspath(
    os.path.join('.', 'models', 'autoencoder'))

image_save_path = os.path.abspath(
    os.path.join('.', 'data', 'epoch_images'))

val_path = os.path.abspath(
    os.path.join('.', 'data', 'rand_trajectories.npy'))
val_data = np.load(val_path)[:128, :, :, :] / 255

def model_builder(hp):
    hp_hidden_layers = hp.Int(
        'hidden_layers', min_value=0, max_value=32, step=4)
    hp_hidden_dim = hp.Int(
        'hidden_dim', min_value=8, max_value=32, step=4)
    hp_convolution_layers = hp.Int(
        'convolution_layers', min_value=0, max_value=4, step=1)
    hp_convolution_size = hp.Int(
        'convolution_size', min_value=2, max_value=6, step=1)
    hp_convolution_filters = hp.Int(
        'convolution_filters', min_value=1, max_value=4, step=1)
    hp_reg_l2 = hp.Choice(
        'reg_l2', values=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 0.0])
    hp_dropout = hp.Choice(
        'dropout', values=[0.0, 0.01, 0.05, 0.1, 0.2, 0.5])
    hp_learning_rate = hp.Choice(
        'learning_rate', values=[1e-2, 1e-3, 1e-4])

    model = autoencoder.Autoencoder(
        val_data[0].shape, 8, hp_hidden_layers, hp_hidden_dim,
        hp_convolution_layers, hp_convolution_size,
        hp_convolution_filters,
        regularizer = tf.keras.regularizers.L2(l2 = hp_reg_l2),
        dropout = hp_dropout)

    model.compile(
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=hp_learning_rate),
        loss = tf.keras.losses.MeanSquaredLogarithmicError(),
        metrics = [tf.keras.metrics.MeanSquaredError(),  # val_mean_squared_error
                   tf.keras.metrics.MeanSquaredLogarithmicError()])  # val_mean_squared_logarithmic_error

    return model

tuner = kt.Hyperband(model_builder,
                     objective = 'val_mean_squared_logarithmic_error',
                     max_epochs = 10,
                     factor = 3,
                     hyperband_iterations = 2,
                     directory = os.path.abspath(
                         os.path.join('.', 'models')),
                     project_name = 'emp_auto_2')

try:
    tuner.search(
        cartpole_generator.CartPoleFrameGen(50),
        epochs = 10,
        validation_data = (val_data, val_data),
        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor = 'val_mean_squared_logarithmic_error',
                patience = 2),
            tf.keras.callbacks.EarlyStopping(
                monitor = 'val_mean_squared_logarithmic_error',
                min_delta = 0.0001,
                patience = 4,
                restore_best_weights = True)])

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(best_hps)

    autoencoder_model = tuner.hypermodel.build(best_hps)
    history = autoencoder_model.fit(
        cartpole_generator.CartPoleFrameGen(50),
        epochs = 1000,
        validation_data = (val_data, val_data),
        callbacks = [
            autoencoder.SaveImageCallback(
                val_data[0], image_save_path),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor = 'val_mean_squared_logarithmic_error',
                patience = 2),
            tf.keras.callbacks.EarlyStopping(
                monitor = 'val_mean_squared_logarithmic_error',
                min_delta = 0.0001,
                patience = 4,
                restore_best_weights = True)],)

except Exception as e:
    print(e)

save_model = input('Save model? (y/n)')
if save_model.strip().lower()[0] == 'y':
    print('Saving...')
    autoencoder_model.save(model_save_path)
else:
    print('Not saving.')
