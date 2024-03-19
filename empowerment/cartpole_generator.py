import gymnasium as gym
import tensorflow as tf
import numpy as np
import os

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

            if terminated or truncated:
                if len(samples) < self.batch_size:
                    samples.append(np.zeros(self.frame_shape))
                observation, info = self.env.reset()

        assert (len(samples) == self.batch_size), 'len(samples) = {}, batch_size = {}'.format(len(samples), self.batch_size)
        samples = np.stack(samples)
        if self.normalize:
            samples = samples / 255

        return samples, samples

    def __len__(self):
        return self.num_batches

class CartPoleStepGen(tf.keras.utils.Sequence):
    def __init__(self, num_batches, encoder_model, batch_size = 32, normalize = True):
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.normalize = normalize
        self.encoder_model = encoder_model

        self.env = gym.make('CartPole-v1', render_mode='rgb_array')
        self.env.reset()
        self.frame_shape = self.env.render().shape
        self.last = self.get_frame_embedding()

    def get_frame_embedding(self, frame = None):
        if not frame is None:
            return self.encoder_model(
                frame[None, :, :, :] / (
                    255 if self.normalize else 1))[0]

        return self.encoder_model(
            self.env.render()[None, :, :, :] / (
                255 if self.normalize else 1))[0]

    def on_epoch_end(self):
        pass

    def __getitem__(self, index):
        inputs = []
        outputs = []

        while len(inputs) < self.batch_size:
            action = self.env.action_space.sample()
            action_vector = np.zeros(self.env.action_space.n)
            action_vector[action] = 1.0

            observation, reward, terminated, truncated, info = self.env.step(action)
            if terminated or truncated:
                inputs.append(np.concatenate((action_vector, self.last)))
                self.last = self.get_frame_embedding(frame = np.zeros(self.frame_shape))  # "Death"
                outputs.append(self.last)

                if len(inputs) < self.batch_size:
                    inputs.append(np.concatenate((action_vector, self.last)))
                    # "Death" transitions to more "death"
                    outputs.append(self.last)

                observation, info = self.env.reset()  # Reborn
                self.last = self.get_frame_embedding()
            else:
                inputs.append(np.concatenate((action_vector, self.last)))
                self.last = self.get_frame_embedding()
                outputs.append(self.last)

        assert (len(inputs) == self.batch_size), 'len(samples) = {}, batch_size = {}'.format(len(samples), self.batch_size)
        assert len(inputs) == len(outputs)

        return np.stack(inputs), np.stack(outputs)

    def __len__(self):
        return self.num_batches

class CartPoleCombinedGen(tf.keras.utils.Sequence):
    def __init__(self, num_batches, batch_size = 32, normalize = True):
        super(CartPoleCombinedGen, self).__init__()
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.normalize = normalize

        self.env = gym.make('CartPole-v1', render_mode = 'rgb_array')
        self.options = {'low': -0.2, 'high': 0.2}
        self.env.reset(options = self.options)
        self.frame_shape = self.env.render().shape
        self.last = self.get_frame()

    def on_epoch_end(self):
        pass

    def get_frame(self):
        return self.env.render() / (
            255 if self.normalize else 1)

    def __getitem__(self, index):
        input_images = []
        input_actions = []
        outputs = []

        while len(input_images) < self.batch_size:
            action = self.env.action_space.sample()
            action_vector = np.zeros(self.env.action_space.n)
            action_vector[action] = 1.0

            observation, reward, terminated, truncated, info = self.env.step(action)
            if terminated or truncated:
                input_images.append(self.last)
                input_actions.append(action_vector)
                new_frame = np.zeros(self.frame_shape)  # "Death"
                outputs.append(np.stack([self.last, new_frame],
                                        axis = 0))
                self.last = new_frame

                if len(input_images) < self.batch_size:
                    input_images.append(self.last)
                    input_actions.append(action_vector)
                    # "Death" transitions to more "death"
                    outputs.append(np.stack([self.last, self.last],
                                            axis = 0))

                observation, info = self.env.reset(options = self.options)  # Reborn
                self.last = self.get_frame()
            else:
                input_images.append(self.last)
                input_actions.append(action_vector)
                new_frame = self.get_frame()
                outputs.append(np.stack([self.last, new_frame],
                                        axis = 0))
                self.last = new_frame

        assert(len(input_images) == self.batch_size),'len(samples) = {}, batch_size = {}'.format(
                len(samples), self.batch_size)
        assert(len(input_images) == len(input_actions))
        assert(len(input_images) == len(outputs))

        return (np.stack(input_images), np.stack(input_actions)), np.stack(outputs)

    def __len__(self):
        return self.num_batches

if __name__ == '__main__':
    cpcg = CartPoleCombinedGen(1, batch_size = 4)
    print(cpcg.__len__())
    print(cpcg.__getitem__(0)[0][0][0].shape)
    print(cpcg.__getitem__(0)[0][1])
    print(cpcg.__getitem__(0)[1].shape)
