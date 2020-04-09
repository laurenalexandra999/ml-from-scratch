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