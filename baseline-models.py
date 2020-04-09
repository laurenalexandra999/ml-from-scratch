"""
Establish baseline performance on a modeling problem to measure
other model performance against it.

Given a problem more challenging than simple regression or classification,
first generate a random prediction algorithm specific to the problem.
Then you can improve on this with a zero rule algorithm.
"""

"""
random prediction algorithm

This algorithm predicts a random outcome. 

- need to store all of the distinct outcome values in the training data
- fix the random number seed prior to employing the algorithm to make 
  certain each time the same set of random numbers is used
"""
from random import seed 
# makes selections over a random range 
from random import randrange

# generate random predictions
def random_prediction_algorithm(train, test):
    # the output value in the training data is the last column for each row
    output_values = [row[-1] for row in train]
    unique = list(set(output_values))
    predicted = list()
    for row in test:
        # obtain a value from unique
        index = randrange(len(unique))
        # add random selection to predicted list
        predicted.append(unique[index])
    return predicted 

seed(1)
train = [[0], [1], [0], [1], [0], [1]]
test = [[None], [None], [None], [None]]
predictions = random_prediction_algorithm(train, test)
print(predictions) 
# [0, 0, 1, 0]

"""
zero rule algorithm

This algorithm uses information about a problem to create one rule
to use when making predictions. 
"""

"""
Classification 

- one rule: predict the most common class. 
  e.g. a dataset has 90 instances of class A and 10 of class B.
  Since class A is the most common, using a zero_rule_classification
  algorithm would secure a baseline accuracy of 90%. 
"""

def zero_rule_classification(train, test):
    output_values = [row[-1] for row in train]
    # get the class value with the highest count in the training data
    prediction = max(set(output_values), key=output_values.count)
    predicted = [prediction for i in range(len(test))]
    return predicted

# example
seed(1)
train = [['0'], ['0'], ['0'], ['0'], ['1'], ['1']]
test = [[None], [None], [None], [None]]
predictions = zero_rule_classification(train, test)
print(predictions)
# ['0', '0', '0', '0']

"""
Regression

- one rule: predict the central tendency (mean or median)
"""

def zero_rule_regression(train, test):
    output_values = [row[-1] for row in train]
    # get mean of output values
    prediction = sum(output_values) / float(len(output_values))
    predicted = [prediction for i in range(len(test))]
    return predicted 

# example
seed(1)
train = [[10], [15], [12], [15], [18], [20]]
test = [[None], [None], [None], [None]]
predictions = zero_rule_regression(train, test)
print(predictions)
# [15.0, 15.0, 15.0, 15.0]