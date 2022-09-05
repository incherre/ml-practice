'''An implementation of the transformer architecture using numpy.'''
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

class FeedForward:
    def __init__(self, weights, biases, activation_function=relu):
        self.weights = weights
        self.biases = biases
        self.activation_function = activation_function

    def forward(self, input_activations):
        linear_part = (self.weights @ input_activations) + self.biases
        return self.activation_function(linear_part)

class AddAndNorm:
    def __init__(self, wrapped_layer, nudge = 1e-7):
        self.wrapped_layer = wrapped_layer
        self.nudge = nudge

    def forward(self, input_activations):
        post_activations = self.wrapped_layer.forward(input_activations)
        added_activations = input_activations + post_activations

        input_size = added_activations.shape[0]  # Shape = (input_size, batch_size)
        layer_mean = np.sum(added_activations, axis = 0) / input_size
        added_activations = added_activations - layer_mean
        layer_std = np.sqrt(np.sum(np.power(added_activations, 2), axis = 0))
        added_activations = added_activations / (layer_std + self.nudge)

        return added_activations

def attention(queries, keys, values, mask = None, mask_value = -1e9):
    key_dim = keys.shape[0]
    pre_softmax = (np.transpose(queries) @ keys) / np.sqrt(key_dim)
    if mask is not None:
        pre_softmax[mask == 0] = mask_value
    post_softmax = np.exp(pre_softmax) / np.sum(np.exp(pre_softmax), axis=0)
    return values @ np.transpose(post_softmax)

class MultiHeadAttention:
    def __init__(self, forward_size, num_heads, query_weights, key_weights, value_weights, final_weights):
        assert(forward_size % num_heads == 0)
        self.forward_size = forward_size
        self.num_heads = num_heads
        self.head_size = forward_size // num_heads

        assert(len(query_weights) == num_heads)
        assert(query_weights[0].shape == (self.head_size, self.head_size))
        self.query_weights = query_weights

        assert(len(key_weights) == num_heads)
        assert(key_weights[0].shape == (self.head_size, self.head_size))
        self.key_weights = key_weights

        assert(len(value_weights) == num_heads)
        assert(value_weights[0].shape == (self.head_size, self.head_size))
        self.value_weights = value_weights

        assert(final_weights.shape == (forward_size, forward_size))
        self.final_weights = final_weights

    def multi_head_attention(self, queries, keys, values, mask = None):
        num_examples = queries.shape[1]
        query_dim = queries.shape[0]
        key_dim = keys.shape[0]
        value_dim = values.shape[0]

        pre_linear_output = np.zeros((self.forward_size, num_examples))
        for head_index in range(self.num_heads):
            lower_head_range = head_index * self.head_size
            upper_head_range = lower_head_range + self.head_size

            post_linear_queries = self.query_weights[head_index] @ queries[lower_head_range:upper_head_range, :]
            post_linear_keys = self.key_weights[head_index] @ keys[lower_head_range:upper_head_range, :]
            post_linear_values = self.value_weights[head_index] @ values[lower_head_range:upper_head_range, :]

            pre_linear_output[lower_head_range:upper_head_range, :] = attention(
                post_linear_queries, post_linear_keys, post_linear_values, mask = mask)
            
        return self.final_weights @ pre_linear_output

class SelfAttention(MultiHeadAttention):
    def __init__(self, forward_size, num_heads, query_weights, key_weights, value_weights, final_weights):
        super().__init__(forward_size, num_heads, query_weights, key_weights, value_weights, final_weights)

    def forward(self, input_activations):
        return self.multi_head_attention(input_activations, input_activations, input_activations)

class EncoderLayer:
    def __init__(self, attention_sublayer, feedforward_sublayer):
        self.attention_sublayer = AddAndNorm(attention_sublayer)
        self.feedforward_sublayer = AddAndNorm(feedforward_sublayer)

    def forward(self, input_activations):
        return self.feedforward_sublayer.forward(self.attention_sublayer.forward(input_activations))

def init_rand(output_width, input_width, mag = 0.1):
    return (np.random.rand(output_width, input_width) * mag * 2) - mag

class Encoder:
    def __init__(self, num_layers, layer_size, num_heads):
        assert(layer_size % num_heads == 0)
        head_size = layer_size // num_heads

        self.layers = []
        for i in range(num_layers):
            attention_sublayer = SelfAttention(
                layer_size, num_heads, [init_rand(head_size, head_size) for i in range(num_heads)],
                [init_rand(head_size, head_size) for i in range(num_heads)],
                [init_rand(head_size, head_size) for i in range(num_heads)],
                init_rand(layer_size, layer_size))
            feedforward_sublayer = FeedForward(init_rand(layer_size, layer_size), init_rand(layer_size, 1))
        self.layers.append(EncoderLayer(attention_sublayer, feedforward_sublayer))

    def forward(self, input_activations):
        temp_activations = input_activations
        for layer in self.layers:
            temp_activations = layer.forward(temp_activations)
        return temp_activations

class DecoderLayer:
    def __init__(self):
        pass

class Decoder:
    def __init__(self):
        pass

if __name__ == '__main__':
    attention_sublayer = SelfAttention(
        8, 2, [np.eye(4), np.eye(4)], [np.eye(4), np.eye(4)], [np.eye(4), np.eye(4)], np.eye(8))
    feedforward_sublayer = FeedForward(np.eye(8), np.array([[0], [0], [0], [0], [0], [0], [0], [0]]))
    test_encoder_layer = EncoderLayer(attention_sublayer, feedforward_sublayer)
    test_input = np.array([[1, 3], [2, 7], [3, 8], [4, 3], [5, 9], [6, 1], [7, 4], [8, 3]])

    print(test_encoder_layer.forward(test_input))

    test_encoder = Encoder(4, 8, 2)
    print(test_encoder.forward(test_input))
