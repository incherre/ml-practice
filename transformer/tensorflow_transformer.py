'''An implementation of the transformer architecture using tensorflow.'''
import tensorflow as tf
import os
import random

def norm(input_activations, nudge = 1e-7):
    # Shape = (batch_size, embed_size, seq_len)
    input_size = input_activations.shape[1] * input_activations.shape[2]

    layer_mean = tf.math.reduce_sum(input_activations, axis = (1, 2)) / input_size
    input_activations = input_activations - layer_mean[:, None, None]
    layer_std = tf.math.sqrt(tf.math.reduce_sum(tf.math.pow(input_activations, 2),
                                                axis = (1, 2)))
    input_activations = input_activations / (layer_std[:, None, None] + nudge)

    return input_activations

def attention(queries, keys, values, mask = None, mask_value = -1e9):
    batch_size = keys.shape[0]
    key_dim = keys.shape[1]
    seq_len = keys.shape[2]

    pre_softmax = (tf.transpose(queries, perm=(0, 2, 1)) @ keys) / tf.math.sqrt(tf.constant(key_dim, dtype='float32'))
    if mask is not None:
        pre_softmax = pre_softmax * mask
        pre_softmax = pre_softmax + (
            tf.cast(mask == 0, tf.float32) * mask_value
        )
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

            heads_stack.append(attention(post_linear_queries,
                                         post_linear_keys,
                                         post_linear_values, mask = mask))
        pre_linear_output = tf.concat(heads_stack, axis=1)
        return self.final_weights @ pre_linear_output

class SelfAttention(MultiHeadAttention):
    def __init__(self, embedding_size, sequence_length, num_heads,
                 query_weights, key_weights, value_weights, final_weights):
        super(SelfAttention, self).__init__(embedding_size, sequence_length, num_heads,
                                            query_weights, key_weights, value_weights,
                                            final_weights)

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
            tokens_stack.append(
                self.linear_sublayer(self.feedforward_sublayer(post_attention[:, :, i])))
        
        return norm(post_attention + tf.stack(tokens_stack, axis=2))

def init_rand(output_width, input_width, mag = 0.1):
    return (tf.random.uniform(shape=(output_width, input_width)) * mag * 2) - mag

def make_encoder_stack(num_layers, layer_size, hidden_layer_size,
                       sequence_length, num_heads, input_embedding_input):
    assert(layer_size % num_heads == 0)
    head_size = layer_size // num_heads

    output = input_embedding_input

    for i in range(num_layers):
        attention_sublayer = SelfAttention(
            layer_size, sequence_length, num_heads,
            [init_rand(head_size, head_size) for i in range(num_heads)],
            [init_rand(head_size, head_size) for i in range(num_heads)],
            [init_rand(head_size, head_size) for i in range(num_heads)],
            init_rand(layer_size, layer_size))

        feedforward_sublayer = tf.keras.layers.Dense(hidden_layer_size, activation='relu')
        linear_sublayer = tf.keras.layers.Dense(layer_size, activation='linear')

        output = EncoderLayer(
            attention_sublayer,
            feedforward_sublayer,
            linear_sublayer)(output)

    return output

class MaskedSelfAttention(MultiHeadAttention):
    def __init__(self, embedding_size, sequence_length, num_heads,
                 query_weights, key_weights, value_weights, final_weights):
        super().__init__(embedding_size, sequence_length, num_heads,
                         query_weights, key_weights, value_weights, final_weights)
        self.mask = tf.constant(
            tf.linalg.band_part(
                tf.ones((sequence_length, sequence_length)),
                -1, 0))

    def call(self, inputs):
        return self.multi_head_attention(inputs, inputs,
                                         inputs, mask = self.mask)

class EmbeddingAttention(MultiHeadAttention):
    def call(self, inputs):
        embedding, other_inputs = inputs
        return self.multi_head_attention(embedding, embedding, other_inputs)

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, masked_attention_sublayer, embedding_sublayer,
                 feedforward_sublayer, linear_sublayer):
        super(DecoderLayer, self).__init__()
        self.masked_attention_sublayer = masked_attention_sublayer
        self.embedding_sublayer = embedding_sublayer
        self.feedforward_sublayer = feedforward_sublayer
        self.linear_sublayer = linear_sublayer

    def call(self, inputs):
        embedding, other_inputs = inputs

        post_attention = norm(other_inputs + self.masked_attention_sublayer(other_inputs))
        post_embedding = norm(post_attention + self.embedding_sublayer(
            [embedding, post_attention]))

        tokens_stack = []
        for i in range(post_embedding.shape[2]):
            tokens_stack.append(
                self.linear_sublayer(self.feedforward_sublayer(post_embedding[:, :, i])))

        return norm(post_embedding + tf.stack(tokens_stack, axis=2))

def make_decoder_stack(num_layers, layer_size, hidden_layer_size,
                       sequence_length, num_heads, input_embedding_input,
                       output_embedding_input):
        assert(layer_size % num_heads == 0)
        head_size = layer_size // num_heads

        output = output_embedding_input

        for i in range(num_layers):
            masked_attention_sublayer = MaskedSelfAttention(
                layer_size, sequence_length, num_heads,
                [init_rand(head_size, head_size) for i in range(num_heads)],
                [init_rand(head_size, head_size) for i in range(num_heads)],
                [init_rand(head_size, head_size) for i in range(num_heads)],
                init_rand(layer_size, layer_size))
            embedding_sublayer = EmbeddingAttention(
                layer_size, sequence_length, num_heads,
                [init_rand(head_size, head_size) for i in range(num_heads)],
                [init_rand(head_size, head_size) for i in range(num_heads)],
                [init_rand(head_size, head_size) for i in range(num_heads)],
                init_rand(layer_size, layer_size))

            feedforward_sublayer = tf.keras.layers.Dense(hidden_layer_size, activation='relu')
            linear_sublayer = tf.keras.layers.Dense(layer_size, activation='linear')

            output = DecoderLayer(
                masked_attention_sublayer,
                embedding_sublayer,
                feedforward_sublayer,
                linear_sublayer)([input_embedding_input, output])

        return output

class PositionalEncodingLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_dimension, sequence_len):
        super(PositionalEncodingLayer, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.sequence_len = sequence_len
        self.encoding = self.positional_encoding()

    def call(self, inputs):
        return inputs + self.encoding

    def positional_encoding(self):
        encoding_array = []
        for dim in range(self.embedding_dimension):
            seq_array = []
            for pos in range(self.sequence_len):
                seq_array.append(self.encoding_at(dim, pos))
            encoding_array.append(seq_array)
        return tf.constant(tf.convert_to_tensor(encoding_array))

    def encoding_at(self, dim, pos):
        if dim % 2 == 0:
            return tf.math.sin(
                pos / tf.math.pow(10000.0, (
                    dim / self.embedding_dimension)))
        else:
            return tf.math.cos(
                pos / tf.math.pow(10000.0, (
                    (dim - 1) / self.embedding_dimension)))

class TiedInverseLinearLayer(tf.keras.layers.Layer):
    def __init__(self, linear_layer):
        super(TiedInverseLinearLayer, self).__init__()

        self.linear_layer = linear_layer

    def call(self, inputs):
        return tf.matmul(inputs, self.linear_layer.weights[0], transpose_b = True)

def make_transformer(vocab_size, num_layers, embed_dim,
                     hidden_layer_dim, max_length, num_heads):
    input_tokens = tf.keras.Input(
        shape=((vocab_size, max_length)), name="in_embedding")

    output_tokens = tf.keras.Input(
        shape=((vocab_size, max_length)), name="out_embedding")

    linear_embedding = tf.keras.layers.Dense(embed_dim, activation='linear')

    positional_encoder = PositionalEncodingLayer(embed_dim, max_length)

    tokens_stack = []
    for i in range(input_tokens.shape[2]):
        tokens_stack.append(linear_embedding(input_tokens[:, :, i]))

    input_embedding = make_encoder_stack(
        num_layers, embed_dim, hidden_layer_dim,
        max_length, num_heads, positional_encoder(tf.stack(tokens_stack, axis=2)))

    tokens_stack = []
    for i in range(output_tokens.shape[2]):
        tokens_stack.append(linear_embedding(output_tokens[:, :, i]))

    output_embedding = make_decoder_stack(
        num_layers, embed_dim, hidden_layer_dim,
        max_length, num_heads, input_embedding,
        positional_encoder(tf.stack(tokens_stack, axis=2)))

    linear_unembedding = TiedInverseLinearLayer(linear_embedding)

    tokens_stack = []
    for i in range(output_embedding.shape[2]):
        tokens_stack.append(linear_unembedding(output_embedding[:, :, i]))

    output = tf.nn.softmax(tf.stack(tokens_stack, axis=2))

    return tf.keras.Model(inputs = [input_tokens, output_tokens], outputs = [output])

PAD = '<pad>'
START = '<start>'
UNKNOWN = '<?>'

def convert_text(text, alphabet, max_length, shift = False):
    # assumes that the first three elements of alphabet are PAD, START, and UNKNOWN
    alphabet_dict = {}
    for index, element in enumerate(alphabet):
        alphabet_dict[element] = index

    indices = [alphabet_dict[PAD]] * max_length
    next_element = 0
    if shift:
        indices[next_element] = alphabet_dict[START]
        next_element += 1

    for character in text:
        if next_element >= max_length:
            break

        if character in alphabet_dict:
            indices[next_element] = alphabet_dict[character]
        else:
            indices[next_element] = alphabet_dict[UNKNOWN]

        next_element += 1

    return tf.one_hot(indices, len(alphabet), axis = 0)

def convert_tokens(tokens, alphabet):
    text_list = []

    for i in range(tokens.shape[1]):
        samples = tf.random.categorical(
            tf.math.log(tokens[None, :, i]), 1)
        text_list.append(alphabet[tf.cast(samples[0][0], tf.int32)])

    return ''.join(text_list)

def gen_random_string(alphabet, max_length):
    char_list = []

    while len(char_list) < max_length:
        new_char = random.choice(alphabet)
        if new_char == PAD:
            break
        if new_char != START:
            char_list.append(new_char)

    return ''.join(char_list)

def gen_copy_task_data(alphabet, num_examples = 100000, max_length = 32):
    x1 = []
    x2 = []
    y = []

    for i in range(num_examples):
        text = gen_random_string(alphabet, max_length)
        x1.append(convert_text(text, alphabet, max_length))
        x2.append(convert_text(text, alphabet, max_length, shift = True))
        y.append(convert_text(text, alphabet, max_length, shift = True))

    return [tf.stack(x1, axis=0), tf.stack(x2, axis=0)], tf.stack(y, axis=0)

if __name__ == '__main__':
    alphabet = [PAD, START, UNKNOWN, 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ']

    transformer_model = make_transformer(len(alphabet), 4, 8, 16, 32, 2)
    transformer_model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
        loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.01))
    # transformer_model.summary()
    x, y = gen_copy_task_data(alphabet)
    transformer_model.fit(x = x,
                          y = y,
                          batch_size = 32,
                          epochs = 1)
    print(convert_tokens(
        transformer_model([[
            convert_text('hello world', alphabet, 32)[None, :, :],
            convert_text('hello world', alphabet, 32, shift = True)[None, :, :]
        ]])[0],
        alphabet))

    input()
