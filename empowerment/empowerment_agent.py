import os
import gymnasium as gym
import numpy as np
import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

autoencoder_model_path = os.path.abspath(
    os.path.join('.', 'models', 'autoencoder_L0322'))

state_transition_model_path = os.path.abspath(
    os.path.join('.', 'models', 'state_transition_model'))

class CartPoleEmpowermentEnv(py_environment.PyEnvironment):
    def __init__(self,
                 autoencoder_model_path,
                 state_transition_model_path,
                 lookahead=10,
                 fanout=10):
        self.lookahead = lookahead
        self.fanout = fanout
        self.env = gym.make('CartPole-v1', render_mode='rgb_array')
        self.autoencoder_model = tf.keras.models.load_model(autoencoder_model_path)
        self.state_transition_model = tf.keras.models.load_model(state_transition_model_path)

        observation_shape = self.autoencoder_model.encoder.output_shape[1:]
        assert(observation_shape == self.state_transition_model.output_shape[1:])

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=1,
            name='action')

        self._observation_spec = array_spec.BoundedArraySpec(
            shape=observation_shape, dtype=np.float64,
            minimum=0.0, maximum=1.0, name='observation')

        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def get_current_state_embedding(self):
        state = self.env.render()[None, :, :, :] / 255
        if self._episode_ended:
            state = np.zeros(state.shape)
        embedding = self.autoencoder_model.encoder(state)[0]
        return embedding

    def estimate_empowerment(self):
        if self._episode_ended:
            return 0.0

        start_state = self.get_current_state_embedding()
        state_stack = np.stack([start_state] * self.fanout)

        for _ in range(self.lookahead):
            rand_actions = []
            for _ in range(self.fanout):
                action = self.env.action_space.sample()
                action_vector = np.zeros(self.env.action_space.n)
                action_vector[action] = 1.0
                rand_actions.append(action_vector)
            rand_actions = np.stack(rand_actions)

            # Estimate the next state of each trajectory in one inference call.
            state_stack = self.state_transition_model(
                np.concatenate((rand_actions, state_stack), axis=1))

        # np.var will flatten by default, which is done explicitly in
        #   the paper: Assistace via Empowerment.
        return np.var(state_stack)

    def _reset(self):
        self.env.reset()
        self._episode_ended = False
        return ts.restart(
            np.array(self.get_current_state_embedding(),
                     dtype=np.float64))

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        action_vector = np.zeros(self.env.action_space.n)
        if action == 0 or action == 1:
            action_vector[action] = 1.0
        else:
            raise ValueError('`action` should be 0 or 1.')

        observation, reward, terminated, truncated, info = self.env.step(action)
        self._episode_ended = terminated or truncated

        empowerment = self.estimate_empowerment()

        if self._episode_ended:
            return ts.termination(
                np.array(self.get_current_state_embedding(),
                         dtype=np.float64),
                empowerment)
        else:
            return ts.transition(
                np.array(self.get_current_state_embedding(),
                         dtype=np.float64),
                reward=empowerment, discount=1.0)

if __name__ == '__main__':
    environment = CartPoleEmpowermentEnv(autoencoder_model_path,
                                         state_transition_model_path)
    utils.validate_py_environment(environment, episodes=5)
