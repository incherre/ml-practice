'''Performs classification using a saved model.'''
import os
from tensorflow import keras
import tensorflow_hub as hub
import numpy as np

model_path = os.path.join('.', 'models', 'sentence_classifier.h5')

classifier_embedding = None
sentence_embed = None

def get_embedding(sentence):
    global sentence_embed

    if sentence_embed is None:
        sentence_embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')

    return np.asarray(sentence_embed([sentence]))

def classify_sentence(sentence):
    global classifier_embedding

    if classifier_embedding is None:
        classifier_embedding = keras.models.load_model(model_path, compile = False)

    embedding = get_embedding(sentence)
    return classifier_embedding.predict(embedding, verbose = 0)[0][0]

if __name__ == '__main__':
    message = 'That would be fun. I left my strategy books at home so I can\'t lend them to you.'
    print(message)
    print(classify_sentence(message))
