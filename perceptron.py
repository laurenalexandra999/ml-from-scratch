"""
the Perceptron algorithm 

- the simplest form of an artificial neural network 
- used for two-class classification problems (0 and 1)
- a model of a single neuron
    - accepts inputs via dendrites which pass the signal to the cell body

- the Perceptron accepts inputs from training data
- inputs are weighted and combined in a linear activation equation
- the activation is transformed into an output value using a transfer function

- step transfer function
    prediction = 1.0 IF activation (greater than or equal to) 0.0 ELSE 0.0
"""

"""
stochastic gradient descent

- the Perceptron uses SGD to update weights/coefficients

- weights are updated each iteration:

    w = w + learning_rate * (expected - predicted) * x

    - w is the weight being optimized 
    - learning rate you need to configure, e.g. 0.01
    - prediction error for the model is (expected - predicted) and is attributed to the weight
    - x is input
"""