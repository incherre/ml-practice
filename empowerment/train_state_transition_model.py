import tensorflow as tf
import numpy as np
import keras_tuner as kt
import os

import cartpole_generator

model_save_path = os.path.abspath(
    os.path.join('.', 'models', 'state_transition_model'))

val_path_x = os.path.abspath(
    os.path.join('.', 'data', 'obs_act.npy'))
val_data_x = np.load(val_path_x)[:128, :]

val_path_y = os.path.abspath(
    os.path.join('.', 'data', 'result.npy'))
val_data_y = np.load(val_path_y)[:128, :]

model_load_path = os.path.abspath(
    os.path.join('.', 'models', 'autoencoder_L0322'))
autoencoder_model = tf.keras.models.load_model(model_load_path)

hp_names = [
    'hidden_layers',
    'hidden_dim',
    'reg_l2',
    'dropout',
    'learning_rate']

def model_builder(hp):
    hp_hidden_layers = hp.Int(
        'hidden_layers', min_value=0, max_value=128, step=4)
    hp_hidden_dim = hp.Int(
        'hidden_dim', min_value=8, max_value=128, step=4)
    hp_reg_l2 = hp.Choice(
        'reg_l2', values=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 0.0])
    hp_dropout = hp.Choice(
        'dropout', values=[0.0, 0.01, 0.05, 0.1, 0.2, 0.5])
    hp_learning_rate = hp.Choice(
        'learning_rate', values=[1e-2, 1e-3, 1e-4])

    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(val_data_x.shape[1],)))
    for i in range(hp_hidden_layers):
        model.add(tf.keras.layers.Dense(hp_hidden_dim,
                                        activation = tf.keras.activations.relu,
                                        kernel_regularizer = tf.keras.regularizers.L2(l2 = hp_reg_l2),
                                        bias_regularizer = tf.keras.regularizers.L2(l2 = hp_reg_l2)))
        model.add(tf.keras.layers.Dropout(hp_dropout))

    model.add(tf.keras.layers.Dense(val_data_y.shape[1],
                                    activation = tf.keras.activations.sigmoid,
                                    kernel_regularizer = tf.keras.regularizers.L2(l2 = hp_reg_l2),
                                    bias_regularizer = tf.keras.regularizers.L2(l2 = hp_reg_l2)))

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
                     project_name = 'stm_auto_1')

try:
    tuner.search(
        cartpole_generator.CartPoleStepGen(50, autoencoder_model.encoder),
        epochs = 10,
        validation_data = (val_data_x, val_data_y),
        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor = 'val_mean_squared_logarithmic_error',
                patience = 4),
            tf.keras.callbacks.EarlyStopping(
                monitor = 'val_mean_squared_logarithmic_error',
                min_delta = 0.0001,
                patience = 8,
                restore_best_weights = True)])

    best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]
    for hp_name in hp_names:
        print('{}: {}'.format(hp_name, best_hps.get(hp_name)))

    state_transition_model = tuner.hypermodel.build(best_hps)
    history = state_transition_model.fit(
        cartpole_generator.CartPoleStepGen(50, autoencoder_model.encoder),
        epochs = 1000,
        validation_data = (val_data_x, val_data_y),
        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor = 'val_mean_squared_logarithmic_error',
                patience = 4),
            tf.keras.callbacks.EarlyStopping(
                monitor = 'val_mean_squared_logarithmic_error',
                min_delta = 0.0001,
                patience = 8,
                restore_best_weights = True)],)

except Exception as e:
    print(e)

save_model = input('Save model? (y/n)')
if save_model.strip().lower()[0] == 'y':
    print('Saving...')
    state_transition_model.save(model_save_path)
else:
    print('Not saving.')
