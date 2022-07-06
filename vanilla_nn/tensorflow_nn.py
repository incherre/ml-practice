'''An implementation of a vanilla neural network using tensorflow.'''
import tensorflow as tf
import numpy as np
import os

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
        nn.append(Layer(tf.Variable(tf.random.normal([output_width, input_width + 1], stddev=mag)),
                        tf.nn.sigmoid))
        return nn
    else:
        nn.append(Layer(tf.Variable(tf.random.normal([hidden_width, input_width + 1], stddev=mag)),
                        tf.nn.relu))

    for i in range(hidden_count):
        nn.append(Layer(tf.Variable(tf.random.normal([hidden_width, hidden_width + 1], stddev=mag)),
                        tf.nn.relu))

    nn.append(Layer(tf.Variable(tf.random.normal([output_width, hidden_width + 1], stddev=mag)),
                    tf.nn.sigmoid))
    return nn

def forward_pass(input_activations, nn):
    '''Performs a forward pass on the provided neural net.'''
    bias_activations = tf.constant(tf.ones([1, input_activations.shape[1]]))

    result = input_activations
    for layer in nn:
        linear_part = layer.weights @ tf.concat([result, bias_activations], 0)
        result = layer.activation(linear_part)

    return result

def train(inputs, labels, nn, epochs = 2000, learning_rate = 0.1, loss_function = tf.keras.losses.BinaryCrossentropy()):
    '''Trains the given neural net on the provided examples.'''
    loss_history = []

    for i in range(epochs):
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        with tf.GradientTape() as gt:
            predicts = forward_pass(inputs, nn)
            loss = loss_function(labels, predicts)

        grads = gt.gradient(loss, [layer.weights for layer in nn])
        optimizer.apply_gradients(zip(grads, [layer.weights for layer in nn]))
    
        loss_history.append(loss.numpy())

    return loss_history

if __name__ == '__main__':
    train_path = os.path.abspath(os.path.join(".", "data", "train.csv"))
    train_data = np.genfromtxt(train_path, delimiter=',')
    train_inputs = tf.constant(train_data[:,:-1].transpose(), dtype=tf.float32)
    train_labels = tf.constant(train_data[:,-1:].transpose(), dtype=tf.float32)

    nn = init_nn(2, 3, 1, 1)
    loss_history = train(train_inputs, train_labels, nn)
    print('loss:', loss_history[::100])

    test_path = os.path.abspath(os.path.join(".", "data", "test.csv"))
    test_data = np.genfromtxt(test_path, delimiter=',')
    test_inputs = tf.constant(test_data[:,:-1].transpose(), dtype=tf.float32)
    test_labels = tf.constant(test_data[:,-1:].transpose(), dtype=tf.float32)
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
