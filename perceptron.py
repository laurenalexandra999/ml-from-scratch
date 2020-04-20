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

"""
train network weights

- estimate weights with stochastic gradient descent
    - learning rate: used to limit the amount each weight is corrected on each update
    - epochs/exposures: the number of times to run through the training data while updating the weights
"""
def train_weights(train, l_rate, n_epoch):
    weights = [0.0 for i in range(len(train[0]))]
    # iterate over each epoch 
    for epoch in range(n_epoch):
        sum_error = 0.0
        # iterate over each row in training data for an epoch
        for row in train:
            prediction = predict(row, weights)
            # expected - predicted 
            error = row[-1] - prediction
            sum_error += error**2
            # bias(t + 1) = bias(t) + learning_rate * (expected(t) - predicted(t))
            weights[0] = weights[0] + l_rate * error 
            # iterate over each weight and update it for a row in an epoch
            for i in range(len(row) - 1):
                # w(t + 1) = w(t) + learning_rate * (expected(t) - predicted(t) * x(t))
                weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
    return weights 

# example
dataset = [[2.7810836,2.550537003,0],
[1.465489372,2.362125076,0],
[3.396561688,4.400293529,0],
[1.38807019,1.850220317,0],
[3.06407232,3.005305973,0],
[7.627531214,2.759262235,1],
[5.332441248,2.088626775,1],
[6.922596716,1.77106367,1],
[8.675418651,-0.242068655,1],
[7.673756466,3.508563011,1]]
# learning rate
l_rate = 0.1
# epochs/exposures 
n_epoch = 5

weights = train_weights(dataset, l_rate, n_epoch)
"""
>epoch=0, lrate=0.100, error=2.000
>epoch=1, lrate=0.100, error=1.000
>epoch=2, lrate=0.100, error=0.000
>epoch=3, lrate=0.100, error=0.000
>epoch=4, lrate=0.100, error=0.000
"""

print(weights)
# [-0.1, 0.20653640140000007, -0.23418117710000003]

# Perceptron algorithm 
def perceptron(train, test, l_rate, n_epoch):
    predictions = list()
    weights = train_weights(train, l_rate, n_epoch)
    for row in test:
        prediction = predict(row, weights)
        predictions.append(prediction)
return(predictions)

# example 
seed(1)
filename = 'data.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])-1):
    str_column_to_float(dataset, i)
    # convert string class to integers (0 or 1)
    str_column_to_int(dataset, len(dataset[0])-1)
# cross-validate with 3 folds
n_folds = 3   
# learning rate       
l_rate = 0.01      
# 500 exposures 
n_epoch = 500       
# evaluate Perceptron algorithm
scores = evaluate_algorithm(dataset, perceptron, n_folds, l_rate, n_epoch)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
"""
Scores: [76.81159420289855, 69.56521739130434, 72.46376811594203]
Mean Accuracy: 72.947%
"""