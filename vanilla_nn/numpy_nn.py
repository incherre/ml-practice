'''An implementation of a vanilla neural network using numpy.'''
import numpy as np
import os

def relu(x, d = False):
    '''ReLU activation function.'''
    if d:
        # Compute partial derivative instead.
        derivative = np.ones(x.shape)
        derivative[x <= 0] = 0
        return derivative
    
    return x * (x > 0)

def sigmoid(x, d = False):
    '''Sigmoid activation function.'''
    if d:
        # Compute partial derivative instead.
        derivative = sigmoid(x)
        return derivative * (1 - derivative)

    return 1 / (1 + np.exp(-x))

class Layer:
    '''A class to represent a layer in a neural network.'''
    def __init__(self, weights, activation):
        self.weights = weights
        self.activation = activation

    def __repr__(self):
        return str(self.weights) + ' w/ ' + self.activation.__name__

def init_nn(input_width, hidden_width, hidden_count, output_width, mag = 0.1):
    '''Returns Layer objects to represent a neural net.'''
    nn = []

    if hidden_count == 0:
        nn.append(Layer((np.random.rand(output_width, input_width + 1) * mag * 2) - mag, sigmoid))
        return nn
    else:
        nn.append(Layer((np.random.rand(hidden_width, input_width + 1) * mag * 2) - mag, relu))

    for i in range(hidden_count):
        nn.append(Layer((np.random.rand(hidden_width, hidden_width + 1) * mag * 2) - mag, relu))

    nn.append(Layer((np.random.rand(output_width, hidden_width + 1) * mag * 2) - mag, sigmoid))
    return nn

def forward_pass(input_activations, nn):
    '''Performs a forward pass on the provided neural net, saving intermediate results.'''
    batch_size = input_activations.shape[1]
    result = [(None, input_activations)]

    for layer in nn:
        linear_part = layer.weights @ np.vstack((result[-1][1], np.ones((1, batch_size))))
        activation = layer.activation(linear_part)
        result.append((linear_part, activation))

    return result

def predict(input_activations, nn):
    '''Returns the neural net's prediction on the input.'''
    return forward_pass(input_activations, nn)[-1][1]

def binary_cross_entropy(true_labels, predicted_labels, d = False, nudge = 1e-7):
    '''Computes the binary cross entropy loss.'''
    assert true_labels.shape == predicted_labels.shape, 'Mismatched shapes in binary cross entropy.'
    assert true_labels.shape[0] == 1, 'Binary cross entropy is only for single unit output.'
    batch_size = true_labels.shape[1]

    if d:
        # Compute partial derivative instead.
        return -(np.divide(true_labels, predicted_labels) - np.divide(1 - true_labels, 1 - predicted_labels))

    true_labels = true_labels.reshape((batch_size,))
    predicted_labels = predicted_labels.reshape((batch_size,))

    positive_part = np.dot(true_labels, np.log(predicted_labels + nudge).transpose())
    negative_part = np.dot((1 - true_labels), np.log(1 - predicted_labels + nudge).transpose())
    return (-1 / batch_size) * (positive_part + negative_part)

def weight_gradient(nn, inference_history, true_labels, loss_function):
    '''Compute the gradients of the weights with respect to the loss function using back prop.'''
    assert true_labels.shape == inference_history[-1][1].shape, 'Mismatched shapes in gradient computation.'
    batch_size = true_labels.shape[1]

    gradients = []
    for layer in nn:
        gradients.append(np.zeros(layer.weights.shape))

    d_activation = loss_function(true_labels, inference_history[-1][1], d = True)

    for index, nn_layer in reversed(list(enumerate(nn))):
        [current_linear_part, current_activation] = inference_history[index + 1]
        [previous_linear_part, previous_activation] = inference_history[index]

        d_linear_part = d_activation * nn_layer.activation(current_linear_part, d = True)
        d_weights = np.dot(d_linear_part, np.vstack((previous_activation, np.ones((1, batch_size)))).transpose()) / batch_size
        gradients[index] += d_weights

        # Don't need to compute the biases separately, since they're treated just like other weights.
        # But unfortunately, need to remove one weight column when propagating backwards.
        d_activation = np.dot(nn_layer.weights[:,:-1].transpose(), d_linear_part)

    return gradients

def update_nn(nn, gradients, learning_rate = 0.1):
    '''Updates the neural net given the provided gradients.'''
    for index, layer in enumerate(nn):
        assert layer.weights.shape == gradients[index].shape, 'Mismatched shapes in parameter update.'
        layer.weights -= gradients[index] * learning_rate

def train(inputs, labels, nn, epochs = 2000, learning_rate = 0.1, loss_function = binary_cross_entropy):
    '''Trains the given neural net on the provided examples.'''
    loss_history = []

    for i in range(epochs):
        batch_result = forward_pass(inputs, nn)
        loss_history.append(loss_function(labels, batch_result[-1][1]))
        gradient = weight_gradient(nn, batch_result, labels, loss_function = loss_function)
        update_nn(nn, gradient, learning_rate = learning_rate)

    return loss_history

if __name__ == '__main__':
    train_path = os.path.abspath(os.path.join(".", "data", "train.csv"))
    train_data = np.genfromtxt(train_path, delimiter=',')
    train_inputs = train_data[:,:-1].transpose()
    train_labels = train_data[:,-1:].transpose()

    nn = init_nn(2, 3, 1, 1)
    loss_history = train(train_inputs, train_labels, nn)
    print('loss:', loss_history[::100])

    test_path = os.path.abspath(os.path.join(".", "data", "test.csv"))
    test_data = np.genfromtxt(test_path, delimiter=',')
    test_inputs = test_data[:,:-1].transpose()
    test_labels = test_data[:,-1:].transpose()
    test_predictions = forward_pass(test_inputs, nn)[-1][1]
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
