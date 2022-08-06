# ml-practice
Practice implementing various ML models to gain familiarity with them and the frameworks used.

## vanilla_nn
This folder contains three implementations of a plain neural network written using NumPy, TensorFlow, and JAX respectively. It's implemented with various degrees of "from-scratch-ness" across those three frameworks; with NumPy being the most from scratch. It's a binary classifier, fully connected, using relu for hidden activations, and sigmoid for output activations. There is also a script to generate some test data.

## sentence_classifier
This folder contains the training recipe for a binary sentence classifier, leveraging the Universal Sentence Encoder from TensorFlow Hub. It deals with an unequal class split by weighting the classes. There is also a tool which mines for samples near the decision boundary of a saved model for future rating; and a tool to rate saved unrated samples.
