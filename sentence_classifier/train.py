'''Trains a binary sentence classifier.'''
import os
import csv
import random
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import tensorflow_addons as tfa
import numpy as np

# Takes a while the first time, before the data is cached.
embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')

# Options:
save_model = True

# Open training and testing data.
trainpath = os.path.abspath(os.path.join('.', 'data', 'train_data.csv'))
trainfile = open(trainpath, 'r', newline='', encoding='utf-8')
traincsv = csv.reader(trainfile)

testpath = os.path.abspath(os.path.join('.', 'data', 'test_data.csv'))
testfile = open(testpath, 'r', newline='', encoding='utf-8')
testcsv = csv.reader(testfile)

# Generate the sentence embeddings.
train_data = []
for message, _, label in traincsv:
    train_data.append([np.asarray(embed([message])[0]), int(label)])
trainfile.close()

test_data = []
for message, _, label in testcsv:
    test_data.append([np.asarray(embed([message])[0]), int(label)])
testfile.close()

# Split the data.
random.shuffle(train_data)
train_labels = np.asarray([example[1] for example in train_data])
train_data = np.asarray([example[0] for example in train_data])

random.shuffle(test_data)
test_labels = np.asarray([example[1] for example in test_data])
test_data = np.asarray([example[0] for example in test_data])

# ML starts here.
def get_model(hidden_layers, hidden_layer_size):
    model = keras.Sequential()
    first_layer = True

    # Hidden layers.
    for i in range(hidden_layers):
        if first_layer:
            model.add(keras.layers.Dense(hidden_layer_size, input_shape=(512,)))
            first_layer = False
        else:
            model.add(keras.layers.Dense(hidden_layer_size))

    # Output layer.
    if first_layer:
        model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid, input_shape=(512,)))
    else:
        model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
        first_layer = False

    return model

epochs = 550
hidden_layers = 10
hidden_layer_size = 512
threshold = 0.5

print('Training for', epochs, 'epochs, with', hidden_layers, 'hidden layers of size', hidden_layer_size)
model = get_model(hidden_layers, hidden_layer_size)
model.summary()  # Display the model.

model.compile(optimizer=tf.compat.v1.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy', tfa.metrics.F1Score(num_classes=1, threshold=threshold)])

# Train the model.
history = model.fit(train_data,
                    train_labels,
                    epochs=epochs,
                    batch_size=512,
                    validation_data=(test_data, test_labels),
                    verbose=1)

results = model.evaluate(test_data, test_labels, verbose=1)

if save_model:
    model.save(os.path.join('.', 'models', 'sentence_classifier.h5'))

print()
print('Accuracy: ', results[1])
print('F1 score: ', results[2][0])

print()
print('Ratio negative: ', 1 - (sum(test_labels) / len(test_labels)))

true_positive = 0
false_negative = 0

cutoff = 0.5
for index, prediction in enumerate(model.predict(test_data)):
    if test_labels[index] and prediction >= cutoff:
        true_positive += 1
    elif test_labels[index] and prediction < cutoff:
        false_negative += 1

print('Recall:', true_positive / (true_positive + false_negative))
input()

