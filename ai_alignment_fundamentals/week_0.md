# Week 0
## Question 1
What are the main similarities and differences between the process of fitting a linear regression to some data, and the process of training a neural network on the same data?
### My original answer
They both use gradient descent to minimize a loss function. They both perform matrix multiplication operations. A linear regression is kind of like one layer of a neural network, with no nonlinearity. The neural network can be much more expressive, but can take longer to train and evaluate, and can be more prone to overfitting.
### Answer co-written with an LLM
Both fitting a linear regression and training a neural network on the same data involve the use of gradient descent to minimize a loss function. Additionally, both approaches rely on matrix multiplication operations. A linear regression can be seen as a single layer of a neural network without any nonlinearity. Neural networks offer greater expressive power due to their ability to incorporate nonlinearity through activation functions. However, this increased expressiveness comes at the cost of longer training and evaluation times, and a higher risk of overfitting.
## Question 2
Explain why the "nonlinearity" in an artificial neuron (e.g. the sigmoid or RELU function) is so important. What would happen if we removed all the nonlinearities in a deep neural network?
### My original answer
The nonlinearity is what allows the neural network to express complex nonlinear functions. Without any nonlinearities, a deep neural network reduces to a simple linear regression model.
### Answer co-written with an LLM
Nonlinearity is crucial in an artificial neuron, such as the sigmoid or RELU function, because it enables a deep neural network to model and express complex nonlinear functions. Without nonlinearity, a deep neural network would reduce to a simple linear regression model, limiting its ability to capture intricate relationships and patterns in the data.
