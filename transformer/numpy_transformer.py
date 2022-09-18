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
    def __init__(self, weights1, biases1, weights2, biases2, activation_function=relu):
        assert(weights1.shape[0] == weights2.shape[1])
        assert(weights1.shape[1] == weights2.shape[0])
        assert(biases1.shape == (weights1.shape[0], 1))
        assert(biases2.shape == (weights2.shape[0], 1))
        self.weights1 = weights1
        self.biases1 = biases1
        self.activation_function = activation_function
        self.weights2 = weights2
        self.biases2 = biases2

    def forward(self, input_activations):
        linear_part = (self.weights1 @ input_activations) + self.biases1
        hidden_activations = self.activation_function(linear_part)
        return (self.weights2 @ hidden_activations) + self.biases2

class AddAndNorm:
    def __init__(self, wrapped_layer, nudge = 1e-7):
        self.wrapped_layer = wrapped_layer
        self.nudge = nudge

    def forward(self, input_activations):
        post_activations = self.wrapped_layer.forward(input_activations)
        added_activations = input_activations + post_activations

        input_size = added_activations.shape[1] * added_activations.shape[2]   # Shape = (batch_size, embed_size, seq_len)
        layer_mean = np.sum(added_activations, axis = (1, 2)) / input_size
        added_activations = added_activations - layer_mean[:, None, None]
        layer_std = np.sqrt(np.sum(np.power(added_activations, 2), axis = (1, 2)))
        added_activations = added_activations / (layer_std[:, None, None] + self.nudge)

        return added_activations

def attention(queries, keys, values, mask = None, mask_value = -1e9):
    batch_size = keys.shape[0]
    key_dim = keys.shape[1]
    seq_len = keys.shape[2]

    pre_softmax = (np.transpose(queries, axes=(0, 2, 1)) @ keys) / np.sqrt(key_dim)
    if mask is not None:
        pre_softmax[:, mask == 0] = mask_value
    post_softmax = np.exp(pre_softmax) / np.sum(
        np.exp(pre_softmax), axis=2)[:, None, :]
    return values @ np.transpose(post_softmax, axes=(0, 2, 1))

class MultiHeadAttention:
    def __init__(self, embedding_size, sequence_length, num_heads, query_weights, key_weights, value_weights, final_weights):
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

    def multi_head_attention(self, queries, keys, values, mask = None):
        assert(len(queries.shape) == 3)
        assert(queries.shape == keys.shape)
        assert(keys.shape == values.shape)
        assert(values.shape[1] == self.embedding_size)
        assert(values.shape[2] == self.sequence_length)
        num_examples = queries.shape[0]

        pre_linear_output = np.zeros((num_examples, self.embedding_size, self.sequence_length))
        for head_index in range(self.num_heads):
            lower_head_range = head_index * self.head_size
            upper_head_range = lower_head_range + self.head_size

            post_linear_queries = self.query_weights[head_index] @ queries[:, lower_head_range:upper_head_range, :]
            post_linear_keys = self.key_weights[head_index] @ keys[:, lower_head_range:upper_head_range, :]
            post_linear_values = self.value_weights[head_index] @ values[:, lower_head_range:upper_head_range, :]

            pre_linear_output[:, lower_head_range:upper_head_range, :] = attention(
                post_linear_queries, post_linear_keys, post_linear_values, mask = mask)
        return self.final_weights @ pre_linear_output

class SelfAttention(MultiHeadAttention):
    def __init__(self, embedding_size, sequence_length, num_heads,
                 query_weights, key_weights, value_weights, final_weights):
        super().__init__(embedding_size, sequence_length, num_heads,
                         query_weights, key_weights, value_weights, final_weights)

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
    def __init__(self, num_layers, layer_size, hidden_layer_size, sequence_length, num_heads):
        assert(layer_size % num_heads == 0)
        head_size = layer_size // num_heads

        self.layers = []
        for i in range(num_layers):
            attention_sublayer = SelfAttention(
                layer_size, sequence_length, num_heads,
                [init_rand(head_size, head_size) for i in range(num_heads)],
                [init_rand(head_size, head_size) for i in range(num_heads)],
                [init_rand(head_size, head_size) for i in range(num_heads)],
                init_rand(layer_size, layer_size))
            feedforward_sublayer = FeedForward(init_rand(hidden_layer_size, layer_size),
                                               init_rand(hidden_layer_size, 1),
                                               init_rand(layer_size, hidden_layer_size),
                                               init_rand(layer_size, 1))
            self.layers.append(EncoderLayer(attention_sublayer, feedforward_sublayer))

    def forward(self, input_activations):
        temp_activations = input_activations
        for layer in self.layers:
            temp_activations = layer.forward(temp_activations)
        return temp_activations

class MaskedSelfAttention(MultiHeadAttention):
    def __init__(self, embedding_size, sequence_length, num_heads,
                 query_weights, key_weights, value_weights, final_weights):
        super().__init__(embedding_size, sequence_length, num_heads,
                         query_weights, key_weights, value_weights, final_weights)
        self.mask = np.tril(np.ones((sequence_length, sequence_length)))

    def forward(self, input_activations):
        return self.multi_head_attention(input_activations, input_activations,
                                         input_activations, mask = self.mask)

class EmbeddingAttention(MultiHeadAttention):
    def __init__(self, embedding_size, sequence_length, num_heads,
                 query_weights, key_weights, value_weights, final_weights):
        super().__init__(embedding_size, sequence_length, num_heads,
                         query_weights, key_weights, value_weights, final_weights)
        self.embedding = None
        self.expected_embedding_shape = (None, embedding_size, sequence_length)

    def forward(self, input_activations):
        assert(self.embedding is not None)
        assert(self.embedding.shape[0] == input_activations.shape[0])
        return self.multi_head_attention(self.embedding, self.embedding,
                                         input_activations)

    def set_embedding(self, embedding):
        assert(self.expected_embedding_shape[1] == embedding.shape[1])
        assert(self.expected_embedding_shape[2] == embedding.shape[2])
        self.embedding = embedding

class DecoderLayer:
    def __init__(self, masked_attention_sublayer, embedding_sublayer, feedforward_sublayer):
        self.masked_attention_sublayer = AddAndNorm(masked_attention_sublayer)
        self.embedding_sublayer_internal_access = embedding_sublayer
        self.embedding_sublayer = AddAndNorm(embedding_sublayer)
        self.feedforward_sublayer = AddAndNorm(feedforward_sublayer)

    def forward(self, embedding, input_activations):
        self.embedding_sublayer_internal_access.set_embedding(embedding)
        return self.feedforward_sublayer.forward(
            self.embedding_sublayer.forward(
                self.masked_attention_sublayer.forward(
                    input_activations)))

class Decoder:
    def __init__(self, num_layers, layer_size, hidden_layer_size, sequence_length, num_heads):
        assert(layer_size % num_heads == 0)
        head_size = layer_size // num_heads

        self.layers = []
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
            feedforward_sublayer = FeedForward(init_rand(hidden_layer_size, layer_size),
                                               init_rand(hidden_layer_size, 1),
                                               init_rand(layer_size, hidden_layer_size),
                                               init_rand(layer_size, 1))
            self.layers.append(DecoderLayer(masked_attention_sublayer, embedding_sublayer, feedforward_sublayer))

    def forward(self, embedding, input_activations):
        temp_activations = input_activations
        for layer in self.layers:
            temp_activations = layer.forward(embedding, temp_activations)
        return temp_activations

class PositionalEncodingLayer:
    def __init__(self, embedding_dimension):
        self.embedding_dimension = embedding_dimension

    def forward(self, input_activations):
        sequence_len = input_activations.shape[2]
        encoding = np.zeros((self.embedding_dimension, sequence_len))
        for position in range(sequence_len):
            encoding[:, position] = self.positional_encoding(position)[:, 0]
        return input_activations + encoding

    def positional_encoding(self, position):
        encoding = np.zeros((self.embedding_dimension, 1))
        for dim in range(self.embedding_dimension):
            if dim % 2 == 0:
                encoding[dim][0] = np.sin(position / np.power(10000, (dim / self.embedding_dimension)))
            else:
                encoding[dim][0] = np.cos(position / np.power(10000, ((dim - 1) / self.embedding_dimension)))
        return encoding

class Transformer:
    def __init__(self, vocab, num_layers, embed_dim, hidden_layer_dim, max_length, num_heads):
        self.vocab = vocab
        self.vocab_lookup = {}
        for i, t in enumerate(vocab):
            self.vocab_lookup[t] = i

        self.embed_dim = embed_dim
        self.max_length = max_length
        self.linear_embedding = init_rand(embed_dim, len(vocab))
        self.positional_encoding = PositionalEncodingLayer(embed_dim)
        self.encoder = Encoder(num_layers, embed_dim, hidden_layer_dim, max_length, num_heads)
        self.decoder = Decoder(num_layers, embed_dim, hidden_layer_dim, max_length, num_heads)

    def forward(self, sequences):
        # sequences is an arraylike of arraylikes of tokens. sequences[batch][position] = token
        batch_size = len(sequences)
        input_embeddings = np.zeros((batch_size, self.embed_dim, self.max_length))
        for batch_num, batch in enumerate(sequences):
            for token_num, token in enumerate(batch):
                if token_num >= self.max_length:
                    continue

                input_embeddings[batch_num, :, token_num] = self.embed_token(token)[:, 0]

        input_embeddings = self.positional_encoding.forward(input_embeddings)
        input_embeddings = self.encoder.forward(input_embeddings)

        output_tokens = [[] for i in range(batch_size)]
        output_embeddings = self.positional_encoding.forward(
            np.zeros((batch_size, self.embed_dim, self.max_length)))
        for i in range(self.max_length - 1):
            decoder_output = self.decoder.forward(input_embeddings, output_embeddings)
            token_probabilities = self.linear_embedding.T @ decoder_output[:, :, i:i+1]
            token_probabilities = np.exp(token_probabilities) / np.sum(
                np.exp(token_probabilities), axis=1)[:, :, None]
            for batch_num in range(batch_size):
                token = np.random.choice(self.vocab, 1, p=token_probabilities[batch_num, :, 0])[0]
                output_tokens[batch_num].append(token)
                output_embeddings[batch_num, :, i+1] += self.embed_token(token)[:, 0]

        return output_tokens

    def embed_token(self, token):
        assert(token in self.vocab_lookup)
        onehot = np.zeros((len(self.vocab), 1))
        onehot[self.vocab_lookup[token]] = 1
        return self.linear_embedding @ onehot
        

if __name__ == '__main__':
    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
                't', 'u', 'v', 'w', 'x', 'y', 'z', ' ']
    test_transformer = Transformer(alphabet, 4, 8, 16, 32, 2)
    batch_predictions = test_transformer.forward(['hello friend', 'knock knock'])
    print(''.join(batch_predictions[0]))
    print(''.join(batch_predictions[1]))
