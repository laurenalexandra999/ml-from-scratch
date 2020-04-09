"""
classification accuracy

accuracy = (correct predictions / total predictions) * 100 

Use when have a small number of class values.
""" 

# arguments: actual and predicted outcomes
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)): 
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0 

# example 
actual = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
predicted = [0, 1, 0, 0, 0, 1, 0, 1, 1, 1]
accuracy = accuracy_metric(actual, predicted)
print(accuracy)
# 80.0

"""
confusion matrix 

This metric shows comparison of all predictions to the actual values.

The matrix consists of
- (rows) counts of predicted class values
- (columns) counts of actual class values

Why use: you can see incorrect predictions and what mistakes were made.
"""

def confusion_matrix(actual, predicted):
    # unique class values 
    unique = set(actual)
    # create list for each value in unique
    matrix = [list() for x in range(len(unique))]
    for i in range(len(unique)):
        # counts in each cell initialized to 0
        matrix[i] = [0 for x in range(len(unique))]
    lookup = dict()
    # mapping - class to integer value
    for i, value in enumerate(unique):
        lookup[value] = i
    for i in range(len(actual)):
        # column values
        x = lookup[actual[i]]
        # row values 
        y = lookup[predicted[i]]
        matrix[y][x] += 1
    return unique, matrix 

# pretty print matrix
def print_confusion_matrix(unique, matrix):
    # print class labels
    print('(A)' + ' '.join(str(x) for x in unique))
    print('(P)---')
    for i, x in enumerate(unique):
        print('%s| %s' % (x, ' '.join(str(x) for x in matrix[i])))

# example
actual = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
predicted = [0, 1, 0, 0, 0, 1, 0, 1, 1, 1]
unique, matrix = confusion_matrix(actual, predicted)
print_confusion_matrix(unique, matrix) 

"""
class labels of 0 and 1. 

correct predictions are diagonal (top left to bottom right).
e.g. Actual 0, Predicted 0. Actual 1, Predicted 1...

correct: 
3 predictions of 0
4 predictions of 1   

 (A)0 1
 (P)---
0| 3 1
1| 2 4
"""

""" 
mean absolute error

the absolute average of the prediction error values.
"""

def MAE_metric(actual, predicted):
    total_error = 0.0
    for i in range(len(actual)): 
        total_error += abs(predicted[i] - actual[i])
    # divide total prediction errors by the number of actual values
    return total_error / float(len(actual))

# example
actual = [0.1, 0.2, 0.3, 0.4, 0.5]
predicted = [0.11, 0.19, 0.29, 0.41, 0.5]
MAE = MAE_metric(actual, predicted)
print(MAE)
#0.007999999999999993