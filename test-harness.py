"""
algorithm test harnesses

A test harness is used to evaluate an algorithm on a dataset.

needs:
- the resampling method to split up the dataset
- the algorithm under evaluation
- the measure of performance 
"""

""" 
train-test test harness 
"""
from random import seed
from random import randrange
from csv import reader

# CSV file load
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file) 
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset 

# string to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

# train/test split 
def train_test_split(dataset, split):
    train = list()
    # e.g. train on 60% of the data
    train_size = split * len(dataset)
    dataset_copy = list(dataset)
    while len(train) < train_size:
        index = randrange(len(dataset_copy))
        # append random selections from dataset copy to training data
        train.append(dataset_copy.pop(index))
    return train, dataset_copy

# measure accuracy
def accuracy_metric(actual, predicted):
    correct = 0 
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

# evaluate a given algorithm on a train/test split
def evaluate_algorithm(dataset, algorithm, split, *args):
    train, test = train_test_split(dataset, split)
    test_set = list()
    for row in test:
        row_copy = list(row)
        row_copy[-1] = None
        test_set.append(row_copy)
    predicted = algorithm(train, test_set, *args)
    actual = [row[-1] for row in test]
    accuracy = accuracy_metric(actual, predicted)
    return accuracy

# let's evaluate a zero rule classification algorithm
def zero_rule_classification(train, test):
    output_values = [row[-1] for row in train]
    prediction = max(set(output_values), key=output_values.count)
    predicted = [prediction for i in range(len(test))]
    return predicted 

# example
seed(1)
filename = 'data.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])):
    str_column_to_float(dataset, i)
# train on 60% of data
split = 0.6 
accuracy = evaluate_algorithm(dataset, zero_rule_classification, split)
print('Accuracy: %.3f%%' % (accuracy)) 
# Accuracy: 67.427%