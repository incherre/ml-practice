import tensorflow as tf
import numpy as np
import os

import combined_stm
import cartpole_generator

model_save_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), 'models',
                 'combined_state_transition_model.keras'))

image_save_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), 'data',
                 'combined_epoch_images'))

val_data_x, val_data_y = cartpole_generator.CartPoleCombinedGen(
    1, batch_size = 128).__getitem__(0)

model = combined_stm.CombinedStateTransitionModel(
    val_data_x[0][0].shape, 8, 2, 16, 1, 7, 8, 4, 16, 2, noise = 0.001,
    regularizer = tf.keras.regularizers.L2(l2 = 1e-6), dropout = 0.05)

model.compile(
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=1e-3),
    loss = tf.keras.losses.MeanSquaredLogarithmicError(),
    metrics = [tf.keras.metrics.MeanSquaredError(),  # val_mean_squared_error
               tf.keras.metrics.MeanSquaredLogarithmicError()])  # val_mean_squared_logarithmic_error
try:
    history = model.fit(
        cartpole_generator.CartPoleCombinedGen(50),
        epochs = 1000,
        validation_data = (val_data_x, val_data_y),
        callbacks = [
            combined_stm.SaveImageCallback(
                (val_data_x[0][:1], val_data_x[1][:1]), image_save_path),
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
    model.save(model_save_path)
else:
    print('Not saving.')

