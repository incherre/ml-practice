'''An implementation of the transformer architecture using tensorflow.'''
import tensorflow as tf
import os

def norm(input_activations, nudge = 1e-7):
    # Shape = (batch_size, embed_size, seq_len)
    input_size = input_activations.shape[1] * input_activations.shape[2]

    layer_mean = tf.math.reduce_sum(input_activations, axis = (1, 2)) / input_size
    input_activations = input_activations - layer_mean[:, None, None]
    layer_std = tf.math.sqrt(tf.math.reduce_sum(tf.math.pow(input_activations, 2), axis = (1, 2)))
    input_activations = input_activations / (layer_std[:, None, None] + nudge)

    return input_activations

def attention(queries, keys, values, mask = None, mask_value = -1e9):
    batch_size = keys.shape[0]
    key_dim = keys.shape[1]
    seq_len = keys.shape[2]

    pre_softmax = (tf.transpose(queries, perm=(0, 2, 1)) @ keys) / tf.math.sqrt(tf.constant(key_dim, dtype='float32'))
    if mask is not None:
        pre_softmax[:, mask == 0] = mask_value
    post_softmax = tf.math.exp(pre_softmax) / tf.math.reduce_sum(
        tf.math.exp(pre_softmax), axis=2)[:, None, :]
    return values @ tf.transpose(post_softmax, perm=(0, 2, 1))

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, embedding_size, sequence_length, num_heads, query_weights, key_weights, value_weights, final_weights):
        super(MultiHeadAttention, self).__init__()

        self.sequence_length = sequence_length

        assert(embedding_size % num_heads == 0)
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.head_size = embedding_size // num_heads

        assert(len(query_weights) == num_heads)
        assert(query_weights[0].shape == (self.head_size, self.head_size))
        self.query_weights = query_weights

        assert(len(key_weights) == num_heads)
        assert(key_weights[0].shape == (self.head_size, self.head_size))
        self.key_weights = key_weights

        assert(len(value_weights) == num_heads)
        assert(value_weights[0].shape == (self.head_size, self.head_size))
        self.value_weights = value_weights

        assert(final_weights.shape == (embedding_size, embedding_size))
        self.final_weights = final_weights

    def build(self, input_shape):
        pass

    def call(self, inputs):
        raise NotImplementedError

    def multi_head_attention(self, queries, keys, values, mask = None):
        assert(len(queries.shape) == 3)
        assert(queries.shape == keys.shape)
        assert(keys.shape == values.shape)
        assert(values.shape[1] == self.embedding_size)
        assert(values.shape[2] == self.sequence_length)

        heads_stack = []
        for head_index in range(self.num_heads):
            lower_head_range = head_index * self.head_size
            upper_head_range = lower_head_range + self.head_size

            post_linear_queries = self.query_weights[head_index] @ queries[:, lower_head_range:upper_head_range, :]
            post_linear_keys = self.key_weights[head_index] @ keys[:, lower_head_range:upper_head_range, :]
            post_linear_values = self.value_weights[head_index] @ values[:, lower_head_range:upper_head_range, :]

            heads_stack.append(attention(post_linear_queries, post_linear_keys, post_linear_values, mask = mask))
        pre_linear_output = tf.concat(heads_stack, axis=1)
        return self.final_weights @ pre_linear_output

class SelfAttention(MultiHeadAttention):
    def __init__(self, embedding_size, sequence_length, num_heads,
                 query_weights, key_weights, value_weights, final_weights):
        super(SelfAttention, self).__init__(embedding_size, sequence_length, num_heads,
                                            query_weights, key_weights, value_weights, final_weights)

    def call(self, inputs):
        return self.multi_head_attention(inputs, inputs, inputs)

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, attention_sublayer, feedforward_sublayer, linear_sublayer):
        super(EncoderLayer, self).__init__()

        self.attention_sublayer = attention_sublayer
        self.feedforward_sublayer = feedforward_sublayer
        self.linear_sublayer = linear_sublayer

    def build(self, input_shape):
        pass

    def call(self, inputs):
        post_attention = norm(inputs + self.attention_sublayer(inputs))

        tokens_stack = []
        for i in range(post_attention.shape[2]):
            tokens_stack.append(self.linear_sublayer(self.feedforward_sublayer(post_attention[:, :, i])))
        
        return norm(post_attention + tf.stack(tokens_stack, axis=2))

def init_rand(output_width, input_width, mag = 0.1):
    return (tf.random.uniform(shape=(output_width, input_width)) * mag * 2) - mag

def add_encoder_stack(model, num_layers, layer_size, hidden_layer_size, sequence_length, num_heads):
    assert(layer_size % num_heads == 0)
    head_size = layer_size // num_heads

    for i in range(num_layers):
        attention_sublayer = SelfAttention(
            layer_size, sequence_length, num_heads,
            [init_rand(head_size, head_size) for i in range(num_heads)],
            [init_rand(head_size, head_size) for i in range(num_heads)],
            [init_rand(head_size, head_size) for i in range(num_heads)],
            init_rand(layer_size, layer_size))

        feedforward_sublayer = tf.keras.layers.Dense(hidden_layer_size, activation='relu')
        linear_sublayer = tf.keras.layers.Dense(layer_size, activation='linear')

        model.add(EncoderLayer(attention_sublayer, feedforward_sublayer, linear_sublayer))

if __name__ == '__main__':
    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
                't', 'u', 'v', 'w', 'x', 'y', 'z', ' ']
    print(alphabet)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(8, 32)))
    add_encoder_stack(model, 4, 8, 16, 32, 2)
    print(model.output_shape)

