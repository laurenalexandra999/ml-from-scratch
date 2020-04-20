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
    prediction = 1.0 IF activation >= 0.0 ELSE 0.0
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

# predict output for row given weights
def predict(row, weights):
    # first weight is the bias which is not responsible for an input value
    activation = weights[0]
    for i in range(len(row) - 1):
        activation += weights[i + 1] * row[i]
    return 1.0 if activation >= 0.0 else 0.0

# example 
dataset = 
# inputs x1, x2
[[2.7810836,2.550537003,0],
[1.465489372,2.362125076,0],
[3.396561688,4.400293529,0],
[1.38807019,1.850220317,0],
[3.06407232,3.005305973,0],
[7.627531214,2.759262235,1],
[5.332441248,2.088626775,1],
[6.922596716,1.77106367,1],
[8.675418651,-0.242068655,1],
[7.673756466,3.508563011,1]]
# bias, weight1, weight2
weights = [-0.1, 0.20653640140000007, -0.23418117710000003]

for row in dataset:
    # activation = (weight1 * x1) + (weight2 * x2) + bias
    prediction = predict(row, weights)
    print("Expected=%d, Predicted=%d" % (row[-1], prediction))

"""
Expected=0, Predicted=0
Expected=0, Predicted=0
Expected=0, Predicted=0
Expected=0, Predicted=0
Expected=0, Predicted=0
Expected=1, Predicted=1
Expected=1, Predicted=1
Expected=1, Predicted=1
Expected=1, Predicted=1
Expected=1, Predicted=1
"""