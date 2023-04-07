import gymnasium as gym
import numpy as np
import os

out_path = os.path.abspath(os.path.join('.', 'data', 'rand_trajectories.npy'))
num_samples = 1000

env = gym.make('CartPole-v1', render_mode='rgb_array')
samples = []

print('Generating...')
observation, info = env.reset()
sample_shape = env.render().shape
while len(samples) < num_samples:
    if len(samples) % (num_samples // 100) == 0:
        print(len(samples))
    samples.append(env.render())

    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        samples.append(np.zeros(sample_shape))
        observation, info = env.reset()

env.close()

print('Saving...')
np.save(out_path, np.stack(samples))
