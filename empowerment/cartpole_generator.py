import gymnasium as gym
import tensorflow as tf
import numpy as np

class CartPoleFrameGen(tf.keras.utils.Sequence):
    def __init__(self, num_batches, batch_size = 32, normalize = True):
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.normalize = normalize

        self.env = gym.make('CartPole-v1', render_mode='rgb_array')
        self.env.reset()
        self.frame_shape = self.env.render().shape

    def on_epoch_end(self):
        pass
    
    def __getitem__(self, index):
        samples = []

        while len(samples) < self.batch_size:
            samples.append(self.env.render())

            action = self.env.action_space.sample()
            observation, reward, terminated, truncated, info = self.env.step(action)

            if terminated or truncated and len(samples) < self.batch_size:
                samples.append(np.zeros(self.frame_shape))
                observation, info = self.env.reset()

        assert(len(samples) == self.batch_size)
        samples = np.stack(samples)
        if self.normalize:
            samples = samples / 255

        return samples, samples
    
    def __len__(self):
        return self.num_batches

if __name__ == '__main__':
    cpfg = CartPoleFrameGen(2)
    print(cpfg.__len__())
    print(cpfg.__getitem__(0))
