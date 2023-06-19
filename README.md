# ml-practice
Practice implementing various ML models to gain familiarity with them and the frameworks used.

## empowerment
An experiment in which I train an RL agent to optimize its own empowerment, estimated from raw frames of pixels. It is loosely inspired by the paper [Assistance via Empowerment](https://arxiv.org/abs/2006.14796).

## ai_alignment_fundamentals
My answers to the exercises and occasionally the discussion prompts from the [AI Safety Fundamentals' Alignment Course](https://aisafetyfundamentals.com/ai-alignment-curriculum), Spring 2022 edition. The most recent update (Feb 2023) to the course does not seem to have exercises anymore.

## transformer
This folder contains two implementations of the transformer architecture. The first is written using NumPy and is only feed forward because I didn't feel like writing all the training gradient code. The second is in TensorFlow, so is trainable via the magic of automatic differentiation.

## sentence_classifier
This folder contains the training recipe for a binary sentence classifier, leveraging the Universal Sentence Encoder from TensorFlow Hub. It deals with an unequal class split by weighting the classes. There is also a tool which mines for samples near the decision boundary of a saved model for future rating; and a tool to rate saved unrated samples.

## vanilla_nn
This folder contains three implementations of a plain neural network written using NumPy, TensorFlow, and JAX respectively. It's implemented with various degrees of "from-scratch-ness" across those three frameworks; with NumPy being the most from scratch. It's a binary classifier, fully connected, using relu for hidden activations, and sigmoid for output activations. There is also a script to generate some test data.