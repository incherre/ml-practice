import gymnasium as gym
import tensorflow as tf
import numpy as np
import os

input_out_path = os.path.abspath(os.path.join('.', 'data', 'obs_act.npy'))
output_out_path = os.path.abspath(os.path.join('.', 'data', 'result.npy'))
num_samples = 1000

model_save_path = os.path.abspath(
    os.path.join('.', 'models', 'autoencoder_L0017'))
model = tf.keras.models.load_model(model_save_path)
model.encoder.summary()

env = gym.make('CartPole-v1', render_mode='rgb_array')
inputs = []
outputs = []

print('Generating...')
observation, info = env.reset()
sample_shape = env.render().shape
last = model.encoder(env.render()[None, :, :, :] / 255)
while len(inputs) < num_samples:
    if num_samples > 100 and len(inputs) % (num_samples // 100) == 0:
        print(len(inputs))

    action = env.action_space.sample()
    action_vector = np.zeros((1, env.action_space.n))
    action_vector[0][action] = 1.0

    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        inputs.append(np.concatenate((action_vector, last), axis = 1))
        last = model.encoder(np.zeros(sample_shape)[None, :, :, :])  # "Death"
        outputs.append(last)

        inputs.append(np.concatenate((action_vector, last), axis = 1))
        # "Death" always transitions to more "death"
        outputs.append(last)

        observation, info = env.reset()  # Reborn
        last = model.encoder(env.render()[None, :, :, :] / 255)
    else:
        inputs.append(np.concatenate((action_vector, last), axis = 1))
        last = model.encoder(env.render()[None, :, :, :] / 255)
        outputs.append(last)

assert len(inputs) == len(outputs)

env.close()

print('Saving...')
np.save(input_out_path, np.stack(inputs))
np.save(output_out_path, np.stack(outputs))
