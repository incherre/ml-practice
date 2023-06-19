import os
import gymnasium as gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tf_agents.environments import py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec, tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.networks import actor_distribution_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.drivers import py_driver
from tf_agents.policies import py_tf_eager_policy

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
        self._timesteps = 0

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

            # Estimate the next state of each trajectory in one
            #   batch inference call for efficiency.
            state_stack = self.state_transition_model(
                np.concatenate((rand_actions, state_stack), axis=1))

        # np.var will flatten by default, which is done explicitly in
        #   the paper: Assistace via Empowerment.
        return np.var(state_stack)

    def _reset(self):
        self._timesteps = 0
        self.env.reset()
        self._episode_ended = False
        return ts.restart(
            np.array(self.get_current_state_embedding(),
                     dtype=np.float64))

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        self._timesteps += 1

        action_vector = np.zeros(self.env.action_space.n)
        if action == 0 or action == 1:
            action_vector[action] = 1.0
        else:
            raise ValueError('`action` should be 0 or 1.')

        numpy_action = action
        if isinstance(numpy_action, tf.Tensor):
            numpy_action = numpy_action.numpy()

        _, _, terminated, truncated, _ = self.env.step(numpy_action)
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

def compute_avg_return(environment, policy, num_episodes=10):
  total_return = 0.0
  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0

    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return

def collect_episode(environment, policy, num_episodes, observer_list):
  driver = py_driver.PyDriver(
    environment,
    py_tf_eager_policy.PyTFEagerPolicy(
      policy, use_tf_function=True),
    observer_list,
    max_episodes=num_episodes)
  initial_time_step = environment.reset()
  driver.run(initial_time_step)

if __name__ == '__main__':
    # Set up environment
    autoencoder_model_path = os.path.abspath(
        os.path.join('.', 'models', 'autoencoder_L0322'))

    state_transition_model_path = os.path.abspath(
        os.path.join('.', 'models', 'state_transition_model'))

    environment = CartPoleEmpowermentEnv(autoencoder_model_path,
                                         state_transition_model_path)

    # Set up agent
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        environment.observation_spec(),
        tensor_spec.from_spec(environment.action_spec()),
        fc_layer_params=(100,))

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    train_step_counter = tf.Variable(0)

    tf_agent = reinforce_agent.ReinforceAgent(
        environment.time_step_spec(),
        environment.action_spec(),
        actor_network=actor_net,
        optimizer=optimizer,
        normalize_returns=True,
        train_step_counter=train_step_counter)
    tf_agent.initialize()

    # Set up data collection
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        tf_agent.collect_data_spec,
        batch_size = 1,
        max_length = 500)

    def expand_trajectory(traj):
        batch = tf.nest.map_structure(lambda t: tf.expand_dims(t, 0), traj)
        replay_buffer.add_batch(batch)
    replay_observer = [expand_trajectory]

    # Some (not all) hyperparameters
    num_eval_episodes = 10
    num_iterations = 250
    collect_episodes_per_iteration = 2
    log_interval = 25
    eval_interval = 50

    # Training loop
    avg_return = compute_avg_return(environment,
                                    tf_agent.policy,
                                    num_eval_episodes)
    returns = [avg_return]

    for _ in range(num_iterations):
        collect_episode(
            environment,
            tf_agent.collect_policy,
            collect_episodes_per_iteration,
            replay_observer)

        iterator = iter(replay_buffer.as_dataset(sample_batch_size=500*collect_episodes_per_iteration))
        trajectories, _ = next(iterator)
        batched_traj = tf.nest.map_structure(
            lambda t: tf.expand_dims(t, axis=0),
            trajectories
        )
        train_loss = tf_agent.train(experience=batched_traj)
        replay_buffer.clear()

        step = tf_agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss.loss))

        if step % eval_interval == 0:
            avg_return = compute_avg_return(environment,
                                            tf_agent.policy,
                                            num_eval_episodes)
            print('step = {0}: Average Return = {1}'.format(step, avg_return))
            returns.append(avg_return)

    steps = range(0, num_iterations + 1, eval_interval)
    plt.plot(steps, returns)
    plt.ylabel('Average Return')
    plt.xlabel('Step')
    plt.show()
