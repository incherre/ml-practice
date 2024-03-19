import tensorflow as tf
import numpy as np
import os
from PIL import Image

import combined_stm
import cartpole_generator

model_save_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), 'models',
                 'combined_state_transition_model.keras'))

image_save_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), 'data',
                 'combined_tinkering_images'))

batch_size = 16
val_data_x, val_data_y = cartpole_generator.CartPoleCombinedGen(
    1, batch_size = batch_size).__getitem__(0)

model = combined_stm.CombinedStateTransitionModel(
    val_data_x[0][0].shape, 8, 2, 16, 1, 7, 8, 4, 16, 2, noise = 0.001,
    regularizer = tf.keras.regularizers.L2(l2 = 1e-6), dropout = 0.05)

model.load_weights(model_save_path)

model.compile(
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=1e-3),
    loss = tf.keras.losses.MeanSquaredLogarithmicError(),
    metrics = [tf.keras.metrics.MeanSquaredError(),  # val_mean_squared_error
               tf.keras.metrics.MeanSquaredLogarithmicError()])  # val_mean_squared_logarithmic_error

output = model(val_data_x)

print(model.encoder(val_data_x))

for i in range(batch_size):
    try:
        image = Image.fromarray(
            np.asarray(
                output[i][0] * 255
            ).astype(np.uint8)
        )
        path = os.path.join(image_save_path,
                            'step_{}_autoencoded.png'.format(i))
        image.save(path)

        image = Image.fromarray(
            np.asarray(
                output[i][1] * 255
            ).astype(np.uint8)
        )
        path = os.path.join(image_save_path,
                            'step_{}_predicted.png'.format(i))
        image.save(path)
    except Exception as e:
        print(e)
