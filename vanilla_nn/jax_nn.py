'''An implementation of a vanilla neural network using jax.'''
from jax import grad, random, tree_util
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np

# Using Colab because jax is lacking windows support.
from google.colab import files

def init_nn(input_width, hidden_width, hidden_count, output_width, mag = 0.1, key = random.PRNGKey(0)):
    '''Returns a list of matrix objects to represent a neural net.'''
    nn = []

    if hidden_count == 0:
        key, subkey = random.split(key)
        nn.append(random.normal(subkey, shape = (output_width, input_width + 1)) * mag)
        return nn
    else:
        key, subkey = random.split(key)
        nn.append(random.normal(subkey, shape = (hidden_width, input_width + 1)) * mag)

    for i in range(hidden_count):
        key, subkey = random.split(key)
        nn.append(random.normal(subkey, shape = (hidden_width, hidden_width + 1)) * mag)

    key, subkey = random.split(key)
    nn.append(random.normal(subkey, shape = (output_width, hidden_width + 1)) * mag)
    return nn

def forward_pass(input_activations, nn, inside_activation = jnn.relu, final_activation = jnn.sigmoid):
    '''Performs a forward pass on the provided neural net.'''
    bias_activations = jnp.ones([1, input_activations.shape[1]])

    result = input_activations
    for layer_weights in nn[:-1]:
        linear_part = layer_weights @ jnp.vstack((result, bias_activations))
        result = inside_activation(linear_part)
    
    linear_part = nn[-1] @ jnp.vstack((result, bias_activations))
    result = final_activation(linear_part)

    return result

def binary_cross_entropy(true_labels, predicted_labels, nudge = 1e-7):
    '''Computes the binary cross entropy loss.'''
    assert true_labels.shape == predicted_labels.shape, 'Mismatched shapes in binary cross entropy.'
    assert true_labels.shape[0] == 1, 'Binary cross entropy is only for single unit output.'
    batch_size = true_labels.shape[1]

    true_labels = true_labels.reshape((batch_size,))
    predicted_labels = predicted_labels.reshape((batch_size,))

    positive_part = jnp.dot(true_labels, jnp.log(predicted_labels + nudge).transpose())
    negative_part = jnp.dot((1 - true_labels), jnp.log(1 - predicted_labels + nudge).transpose())
    return (-1 / batch_size) * (positive_part + negative_part)

def train(inputs, labels, nn, epochs = 2000, learning_rate = 0.1, loss_function = binary_cross_entropy):
    '''Trains the given neural net on the provided examples.'''
    loss_history = []

    grad_function = grad(
        lambda nn, inputs, labels:
        loss_function(labels, forward_pass(inputs, nn))
    )

    for i in range(epochs):
        loss = loss_function(labels, forward_pass(inputs, nn))
        loss_history.append(float(loss.to_py()))

        gradients = grad_function(nn, inputs, labels)
        for index, layer in enumerate(nn):
            nn[index] = layer - (gradients[index] * learning_rate)

    return loss_history

if __name__ == '__main__':
    # Running on colab, so need to upload the files before they can be read.
    uploaded = files.upload()

    train_path = "train.csv"
    train_data = np.genfromtxt(train_path, delimiter=',')
    train_inputs = train_data[:,:-1].transpose()
    train_labels = train_data[:,-1:].transpose()

    nn = init_nn(2, 3, 1, 1)
    loss_history = train(train_inputs, train_labels, nn)
    print('loss:', loss_history[::100])

    test_path = "test.csv"
    test_data = np.genfromtxt(test_path, delimiter=',')
    test_inputs = test_data[:,:-1].transpose()
    test_labels = test_data[:,-1:].transpose()
    test_predictions = forward_pass(test_inputs, nn)
    tp = sum([1 for i in range(len(test_labels[0]))
              if test_labels[0][i] >= 0.5 and test_predictions[0][i] >= 0.5])
    fp = sum([1 for i in range(len(test_labels[0]))
              if test_labels[0][i] < 0.5 and test_predictions[0][i] >= 0.5])
    tn = sum([1 for i in range(len(test_labels[0]))
              if test_labels[0][i] < 0.5 and test_predictions[0][i] < 0.5])
    fn = sum([1 for i in range(len(test_labels[0]))
              if test_labels[0][i] >= 0.5 and test_predictions[0][i] < 0.5])
    print('accuracy:', tp / (tp + fp))
    print('recall:', tp / (tp + fn))
